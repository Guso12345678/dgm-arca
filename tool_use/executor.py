"""The tool-use workflow: the five steps from class, implemented by hand, with no framework.

    1. Get the user message together with the system prompt that carries the tool definitions.
    2. The model generates: text, or one or more <tool_call> blocks.
    3. The application executes the calls (``ToolRegistry.call``) and never trusts the
       arguments blindly: they are validated first.
    4. Each result goes back into the conversation as a message with role ``tool``.
    5. The model is prompted again; if it answers without calling tools, we are done.

Steps 2-4 repeat until the model answers or ``max_turns`` is reached (multi-step tool use).
Several <tool_call> blocks in one generation are executed in the same turn (parallel calls).

``run_tool_loop`` is what ``POST /tools`` calls. Until you implement it, the endpoint
answers 501.

Decisions worth defending in the report:

* Everything that goes wrong becomes an *observation* the model can react to, never an
  exception: malformed JSON, arguments that fail validation, unknown tools, tools that
  raise, repeated calls and actions that were not confirmed.
* Tools registered with ``requires_confirmation=True`` go through ``self.confirm`` before
  they run. In this phase the user's own request is the confirmation (policy ``auto``),
  because ``POST /tools`` is a single non-interactive call; policy ``deny`` blocks them.
  Phase 4 plugs a human-in-the-loop callback into the same hook.
* Decoding is greedy (``do_sample=False``) so the benchmark is reproducible. Thinking is
  off by default: for tool selection it costs latency and the benchmark measures whether
  the extra reasoning pays off (an ablation, see ``ARCA_TOOLS_THINKING``).
* A call that repeats an earlier successful one (same tool, same arguments) is not run
  again: the model is reminded of the result. It is a loop guard, and it saves a request.

Environment variables (all optional): ``ARCA_TOOLS_MODEL``, ``ARCA_TOOLS_THINKING`` (0/1),
``ARCA_TOOLS_CONFIRM`` (auto/deny), ``ARCA_TOOLS_GUIDELINES`` (0/1), ``ARCA_TOOLS_FEWSHOT``
(0/1), ``ARCA_TOOLS_MAX_NEW_TOKENS``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.parser import ToolCall, format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

log = logging.getLogger("arca.tools")

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant designed to help users.
You will be given a task to solve. To help you, you have access to a set of tools.
Each tool is a function and has a description explaining what it does and the inputs it
expects:
---
{tools_description}
---
You should decide if you want to call a tool. If so, your call must be formatted as follows:
<tool_call>
{{"name": "tool name", "arguments": {{"argument": "value"}}}}
</tool_call>
You may emit several <tool_call> blocks in one message if the calls are independent.
After the tool calls are executed, you will receive their results and can continue.
When you have everything you need, answer the user directly without calling tools."""

# Domain rules appended to the generic template. They are an ablation target in the report
# (with / without), so they live in their own constant.
DOMAIN_GUIDELINES = "\n".join(
    [
        "",
        "Additional rules:",
        "- Reply in the same language as the user.",
        "- Never calculate loan figures (instalment, total cost, APR/TAE, debt ratio) by hand: "
        "call loan_metrics.",
        "- Never emit a call whose arguments depend on the result of another call in the same "
        "message. If the rate is 'Euribor plus a spread', emit ONLY the get_euribor call and "
        "stop; after you receive its result, call loan_metrics with annual_rate_pct = "
        "Euribor + spread.",
        "- Call save_credit_report only if the user explicitly asks to save or file a report.",
        "- If the question is conceptual and needs no data or calculation, answer directly "
        "without calling any tool.",
    ]
)

# A worked example of the dependent chain, with numbers that are NOT in the benchmark.
# Small models imitate demonstrations far better than they follow rules, so this is its own
# ablation (``ARCA_TOOLS_FEWSHOT``).
FEWSHOT_EXAMPLE = "\n".join(
    [
        "",
        "Example of a dependent chain (follow this order, one step per message):",
        "User: Quiero 15000 euros a 48 meses al Euribor a 3 meses más 3 puntos. ¿Cuál es la cuota?",
        "Assistant:",
        "<tool_call>",
        '{"name": "get_euribor", "arguments": {"tenor": "3M"}}',
        "</tool_call>",
        'Tool result: {"tenor": "3M", "period": "2026-01", "rate_pct": 2.1, "source": "ECB"}',
        "Assistant:",
        "<tool_call>",
        '{"name": "loan_metrics", "arguments": {"principal": 15000, "annual_rate_pct": 5.1, '
        '"months": 48}}',
        "</tool_call>",
        "(2.1 + 3 = 5.1. Write the sum already computed: never an expression such as "
        "2.1 + 3 inside the JSON.)",
    ]
)

PARSE_ERROR_HINT = (
    " A tool_call must be ONE JSON object with literal values: compute sums yourself "
    "(write 5.247, not 3.247 + 2), and never call a tool with arguments that depend on "
    "results you have not received yet."
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def build_system_prompt(
    registry: ToolRegistry, guidelines: bool = True, fewshot: bool = False
) -> str:
    """Step 1: the system prompt with the tool definitions (and, optionally, domain rules)."""
    prompt = SYSTEM_PROMPT_TEMPLATE.format(tools_description=registry.render_for_prompt())
    if guidelines:
        prompt += DOMAIN_GUIDELINES
        if fewshot:
            prompt += FEWSHOT_EXAMPLE
    return prompt


def confirm_from_env() -> Callable[[str, dict], bool]:
    """Confirmation policy for actions with side effects, from ``ARCA_TOOLS_CONFIRM``."""
    policy = os.environ.get("ARCA_TOOLS_CONFIRM", "auto").strip().lower()
    if policy == "deny":
        return lambda name, arguments: False
    return lambda name, arguments: True


def _call_key(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)


class ToolLoop:
    """Holds the model and the registry and runs the five-step workflow."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        model_id: str | None = None,
        confirm: Callable[[str, dict], bool] | None = None,
        enable_thinking: bool | None = None,
        guidelines: bool | None = None,
        fewshot: bool | None = None,
        max_new_tokens: int | None = None,
    ):
        self.registry = registry or default_registry()
        self.model_id = model_id or os.environ.get("ARCA_TOOLS_MODEL", "Qwen/Qwen3-1.7B")
        self.confirm = confirm or confirm_from_env()
        self.enable_thinking = (
            _env_flag("ARCA_TOOLS_THINKING", False) if enable_thinking is None else enable_thinking
        )
        self.guidelines = (
            _env_flag("ARCA_TOOLS_GUIDELINES", True) if guidelines is None else guidelines
        )
        self.fewshot = _env_flag("ARCA_TOOLS_FEWSHOT", True) if fewshot is None else fewshot
        self.max_new_tokens = max_new_tokens or int(
            os.environ.get("ARCA_TOOLS_MAX_NEW_TOKENS", "1024")
        )
        self.model = None
        self.tokenizer = None
        self.last_messages: list[dict] = []  # the full conversation of the last run, for debugging

    # -------------------------------------------------------------------- model

    def load(self) -> None:
        """Load tokenizer and model (bf16 on GPU).

        The tool definitions are pasted into the system prompt (``build_system_prompt``)
        instead of passing ``tools=`` to ``apply_chat_template``, so what the model sees is
        exactly the template from the slides, and the ablations are about our own text.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()

    def generate(self, messages: list[dict]) -> str:
        """Step 2: one greedy generation from the conversation so far."""
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # --------------------------------------------------------------------- loop

    def _execute(
        self, call: ToolCall, done: dict[str, ToolCallRecord] | None = None
    ) -> ToolCallRecord:
        """Step 3 for one call. Never raises: every failure is an observation."""
        if call.error is not None:
            return ToolCallRecord(
                name=call.name or "<invalid>",
                arguments={},
                ok=False,
                error=call.error + PARSE_ERROR_HINT,
            )
        previous = (done or {}).get(_call_key(call.name, call.arguments))
        if previous is not None:
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=(
                    f"duplicate call: {call.name} was already called with these arguments and "
                    f"returned {format_tool_result(call.name, previous.result)}. Use that "
                    "result instead of calling it again."
                ),
            )
        tool = self.registry.tools.get(call.name)
        if tool is not None and tool.requires_confirmation:
            approved = self.confirm(call.name, call.arguments)
            log.info("confirmation for %s %s: %s", call.name, call.arguments, approved)
            if not approved:
                return ToolCallRecord(
                    name=call.name,
                    arguments=call.arguments,
                    ok=False,
                    error=(
                        "confirmation required: the user has not approved this action, so it "
                        "was NOT executed. Tell the user what you would do and ask them to "
                        "confirm."
                    ),
                )
        return self.registry.call(call.name, call.arguments)

    def run(self, query: str, max_turns: int = 5) -> ToolsResponse:
        """Steps 1-5: loop until the model answers without calling tools or turns run out."""
        system = build_system_prompt(self.registry, self.guidelines, self.fewshot)
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        self.last_messages = messages
        records: list[ToolCallRecord] = []
        done: dict[str, ToolCallRecord] = {}  # successful calls, to catch repeats
        last_text = ""

        for turn in range(max_turns):
            raw = self.generate(messages)
            parsed = parse_tool_calls(raw)

            if not parsed.tool_calls:  # step 5: a plain answer ends the loop
                return ToolsResponse(
                    answer=parsed.text or raw.strip(),
                    tool_calls=records,
                    turns=turn + 1,
                    model=self.model_id,
                )

            last_text = parsed.text
            messages.append({"role": "assistant", "content": raw})
            for call in parsed.tool_calls:  # parallel calls: all of this generation, in order
                record = self._execute(call, done)  # step 3
                records.append(record)
                if record.ok:
                    done[_call_key(record.name, record.arguments)] = record
                payload = record.result if record.ok else f"ERROR: {record.error}"
                messages.append(  # step 4
                    {
                        "role": "tool",
                        "name": record.name,
                        "content": format_tool_result(record.name, payload),
                    }
                )

        # Out of turns: return what we have and say so, instead of looping forever.
        answer = last_text or "No he podido completar la petición en el número máximo de turnos."
        return ToolsResponse(
            answer=answer, tool_calls=records, turns=max_turns, model=self.model_id
        )


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    """Create and load the loop once; the API keeps the instance."""
    global _loop
    if _loop is None:
        loop = ToolLoop()
        loop.load()
        _loop = loop
    return _loop


def run_tool_loop(query: str, max_turns: int = 5) -> ToolsResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_tool_loop().run(query, max_turns=max_turns)


__all__ = [
    "DOMAIN_GUIDELINES",
    "FEWSHOT_EXAMPLE",
    "ToolCallRecord",
    "ToolLoop",
    "build_system_prompt",
    "build_tool_loop",
    "run_tool_loop",
]