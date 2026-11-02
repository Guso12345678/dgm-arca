"""The ReAct agent: Thought -> Action -> Observation, in a loop, with every tool we built.

This is where phases 1 to 3 meet. The brain is a model (``brain="base"``, our phase-1 RLM
with ``brain="rlm"``, or a thinking model with ``brain="thinking"``); the hands are the
phase-2 registry plus ``search_knowledge_base`` from phase 3; the loop is ours.

``ReActAgent.run`` is what ``POST /agent`` calls. Decisions worth defending in the report:

* **One action per step.** If the model emits several ``<tool_call>`` blocks, only the first
  one runs and it is told so. Phase 2 showed that small models write dependent calls in one
  message (the Euribor and the loan that needs it) and invent the value of the first one.
* **Everything that goes wrong is an observation**, never an exception: malformed JSON,
  arguments that fail validation, unknown tools, tools that raise, tools that time out, tools
  that need a confirmation nobody gave, and repeated calls.
* **Per-tool timeout** (``tool_timeout_s``, longer for the knowledge base, whose first call
  may load an index). The tool runs in a daemon thread: Python cannot kill a thread, but a
  stuck tool cannot block the agent, and the model gets an observation to react to.
* **Loop guard.** The second identical call is not executed (the model is reminded of the
  result); the third ends the run as failed instead of burning the step budget.
* **Prose instead of ``final_answer``.** If the model answers in plain text after using at
  least one tool, that text is accepted as the final answer (the step is flagged). If it has
  not used any tool yet it gets one reminder first: a model that answers a calculation from
  memory is exactly what the tools are for.
* **Confirmations.** Tools with ``requires_confirmation`` go through ``self.confirm``. As in
  phase 2 the policy is ``auto`` (``ARCA_AGENT_CONFIRM=deny`` blocks them): ``POST /agent`` is
  a single non-interactive call, so the user's task is the confirmation.
* **Memory** (see ``agent/memory.py``): the task is pinned, observations are clipped, and the
  oldest turns are summarised by extraction only when the token budget is exceeded.
* **Brains are interchangeable and lazy.** ``base`` and ``rlm`` share one Qwen3-0.6B in
  memory (the LoRA adapter is switched off for ``base``); ``thinking`` is a Qwen3-1.7B with
  thinking on. Nothing is loaded until a brain is used.

Environment variables (all optional): ``ARCA_RLM_BASE_MODEL``, ``ARCA_RLM_ADAPTER``,
``ARCA_THINKING_MODEL``, ``ARCA_AGENT_CONFIRM`` (auto/deny), ``ARCA_AGENT_SUMMARY``
(extractive/llm), ``ARCA_AGENT_RETRIEVER`` (bm25/dense/hybrid), ``ARCA_AGENT_PRELOAD_KB``
(0/1), ``ARCA_AGENT_FIXTURE`` (``bench`` freezes the Euribor so the benchmark is reproducible).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent.memory import ConversationMemory, clip, extractive_summary
from agent.prompts import REACT_SYSTEM_PROMPT, SUMMARY_PROMPT
from api.schemas import AgentAction, AgentResponse, AgentStep, ToolCallRecord
from tool_use.executor import PARSE_ERROR_HINT, confirm_from_env
from tool_use.parser import ToolCall, format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry, tool
from tool_use.tools import default_registry

log = logging.getLogger("arca.agent")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_THINKING_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_CANDIDATES = ("rlm/weights/grpo_lora", "rlm/weights/sft_lora")
MAX_NEW_TOKENS = {"base": 512, "rlm": 512, "thinking": 1536}

# The Euribor used when ``ARCA_AGENT_FIXTURE=bench``: the real tool reads the ECB, so its
# value changes every month and a benchmark with a numeric check would stop being valid.
BENCH_EURIBOR = {"1M": 2.0, "3M": 2.1, "6M": 2.2, "12M": 2.5}
BENCH_EURIBOR_PERIOD = "2026-10"

AGENT_RULES = "\n".join(
    [
        "",
        "Additional rules:",
        "- Reply in the same language as the user.",
        "- Emit ONE tool call per message, inside <tool_call></tool_call>, and stop: the "
        "observation is written by the system, never by you. Arguments are plain JSON values "
        "(write 5.5, not 2.5 + 3).",
        "- Never calculate loan figures (instalment, total cost, APR/TAE, debt ratio) by hand: "
        "call loan_metrics.",
        "- If the rate is 'Euribor plus a spread', call get_euribor first; when you receive "
        "it, call loan_metrics with annual_rate_pct = Euribor + spread, already added up.",
        "- For questions about consumer-credit regulation (Ley 16/2011, Directiva (UE) "
        "2023/2225) call search_knowledge_base and rely on the passages it returns; do not "
        "answer legal facts from memory. If the passages do not contain the answer, say so.",
        "- Call save_credit_report only if the user explicitly asks to save or file a report, "
        "and after the figures it should contain have been computed.",
        "- If the task cannot be done with these tools or this data, say so clearly in "
        "final_answer instead of inventing a result.",
    ]
)

NUDGE = (
    "Reminder: you have not used any tool yet. Reply with exactly one <tool_call> block. "
    "Use the tools to get the data or do the calculation, and finish by calling "
    "final_answer. If the task cannot be done with the tools, call final_answer and say so."
)


# ---------------------------------------------------------------------------------------
# The two tools the agent adds to the phase-2 registry
# ---------------------------------------------------------------------------------------


@tool
def final_answer(answer: str) -> str:
    """Return the final answer to the user and stop the loop.

    Args:
        answer: The complete answer to the task.
    """
    return answer


_kb_search: Callable[[str, int], list] | None = None


def set_kb_search(function: Callable[[str, int], list] | None) -> None:
    """Replace the retrieval function (tests, notebooks). ``None`` restores the default."""
    global _kb_search
    _kb_search = function


def _default_kb_search(query: str, top_k: int) -> list:
    from rag.generate import build_retrieval_pipeline

    retriever = os.environ.get("ARCA_AGENT_RETRIEVER", "hybrid")
    return build_retrieval_pipeline().retrieve(query, top_k, retriever)


def render_hits(hits: list, max_chars: int = 700) -> str:
    """Hits as ``[id] (source)`` plus the text, clipped: what the model reads."""
    blocks = []
    for hit in hits:
        text = clip(" ".join(hit.chunk.text.split()), max_chars)
        blocks.append(f"[{hit.chunk.id}] ({hit.chunk.source})\n{text}")
    return "\n\n".join(blocks)


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search the consumer-credit regulation (Ley 16/2011 and Directiva 2023/2225) and return
    the most relevant passages with their ids.

    Args:
        query: What to look for, phrased as a question in Spanish.
        top_k: How many passages to return (1 to 10).
    """
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    hits = (_kb_search or _default_kb_search)(query, top_k)
    return render_hits(hits) if hits else "No passages found for this query."


BRAINS = {
    "base": lambda: os.environ.get("ARCA_RLM_BASE_MODEL", DEFAULT_BASE_MODEL),
    "rlm": lambda: rlm_adapter_path() or "",
    "thinking": lambda: os.environ.get("ARCA_THINKING_MODEL", DEFAULT_THINKING_MODEL),
}


def rlm_adapter_path() -> str | None:
    """The phase-1 adapter: ``ARCA_RLM_ADAPTER``, else the GRPO one, else the SFT one."""
    configured = os.environ.get("ARCA_RLM_ADAPTER", "").strip()
    for candidate in [configured] if configured else list(ADAPTER_CANDIDATES):
        if candidate and (Path(candidate) / "adapter_config.json").exists():
            return candidate
    return None


def freeze_euribor(registry: ToolRegistry, rates: dict[str, float] | None = None) -> None:
    """Replace ``get_euribor`` by a fixed table (same name, schema and description)."""
    table = rates or BENCH_EURIBOR
    real = registry.tools["get_euribor"]

    def fixed(tenor: str = "12M") -> dict:
        return {
            "tenor": tenor,
            "period": BENCH_EURIBOR_PERIOD,
            "rate_pct": table[tenor],
            "source": "fixture",
        }

    registry.tools["get_euribor"] = replace(real, fn=fixed)


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------


def call_with_timeout(function: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``function`` in a daemon thread; raise ``TimeoutError`` if it takes too long."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = function()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"no result after {timeout_s:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _call_key(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)


_FABRICATED_OBSERVATION = re.compile(r"^\s*(?:Observation|Observación)\s*:", re.MULTILINE)
_ACTION_LINE = re.compile(r"^[ \t]*Action:[ \t]*([A-Za-z_]\w*)?[ \t]*\n?[ \t]*(?=\{)", re.MULTILINE)
_SIMPLE_SUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([+-])\s*(\d+(?:\.\d+)?)(?![\w.])")


def _repair_sums(body: str) -> str:
    """``"annual_rate_pct": 2.5 + 3`` is not JSON; the model means 5.5. Add them up."""

    def add(match: re.Match) -> str:
        a, sign, b = float(match.group(1)), match.group(2), float(match.group(3))
        return f"{round(a + b if sign == '+' else a - b, 6):g}"

    return _SIMPLE_SUM.sub(add, body)


def clean_model_output(raw: str) -> str:
    """Make a raw completion parseable without changing what the model decided to do.

    Small models imitate the ReAct transcript and write the *observation* too, then carry on
    as if they had it: everything after the first ``Observation:`` is invented, so it is cut.
    ``Action: {json}`` (and ``Action: name {args}``) is rewritten as a ``<tool_call>`` block,
    and a sum inside the JSON (``2.5 + 3``) is added up. A tool call that is still invalid
    reaches the loop as an error the model can read.
    """
    head, sep, tail = raw.rpartition("</think>")
    match = _FABRICATED_OBSERVATION.search(tail)
    if match:
        tail = tail[: match.start()]
    text = head + sep + tail
    if "<tool_call>" not in text:
        decoder = json.JSONDecoder()
        out, last = [], 0
        for found in _ACTION_LINE.finditer(text):
            if found.start() < last:
                continue
            try:
                payload, end = decoder.raw_decode(text, found.end())
            except json.JSONDecodeError:
                end = text.find("\n", found.end())
                end = len(text) if end < 0 else end
                payload = None
            body = text[found.end() : end]
            if payload is not None and found.group(1):
                body = json.dumps({"name": found.group(1), "arguments": payload})
            out += [text[last : found.start()], f"<tool_call>\n{body}\n</tool_call>"]
            last = end
        text = "".join(out) + text[last:]

    def repair(block: re.Match) -> str:
        body = block.group(1)
        try:
            json.loads(body)
        except json.JSONDecodeError:
            body = _repair_sums(body)
        return f"<tool_call>{body}</tool_call>"

    return re.sub(r"<tool_call>(.*?)</tool_call>", repair, text, flags=re.DOTALL)


def _thought(parsed) -> str:
    text = (parsed.thinking or parsed.text).strip()
    for prefix in ("Thought:", "Pensamiento:"):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _prose_answer(parsed) -> str:
    """The plain-text answer of a message without tool calls ('' if it is a cut-off think)."""
    text = parsed.text.strip()
    if "<think>" in text:  # an unterminated think block is not an answer
        return ""
    for prefix in ("Thought:", "Pensamiento:", "Final answer:", "Respuesta:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


# ---------------------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------------------


@dataclass
class ReActAgent:
    registry: ToolRegistry = field(default_factory=default_registry)
    token_budget: int = 6000
    steps: list[AgentStep] = field(default_factory=list)  # the steps of the last run
    tool_timeout_s: float = 20.0
    kb_timeout_s: float = 300.0
    max_observation_chars: int = 2000
    confirm: Callable[[str, dict], bool] = field(default_factory=confirm_from_env)
    summary_mode: str = field(
        default_factory=lambda: os.environ.get("ARCA_AGENT_SUMMARY", "extractive")
    )

    def __post_init__(self) -> None:
        self._lock = threading.Lock()  # one generation at a time on the shared models
        self._small: tuple | None = None  # (tokenizer, model) of the 0.6B student
        self._small_peft = None  # the same model wrapped with the phase-1 adapter
        self._thinking: tuple | None = None

    @classmethod
    def from_env(cls) -> ReActAgent:
        agent = cls()
        agent.registry.register(final_answer, search_knowledge_base)
        if os.environ.get("ARCA_AGENT_FIXTURE", "").strip().lower() == "bench":
            freeze_euribor(agent.registry)
        agent.load()
        return agent

    def load(self) -> None:
        """Warm up the knowledge base. The brains load on first use (they are heavy and only
        the one being asked for is needed)."""
        if os.environ.get("ARCA_AGENT_PRELOAD_KB", "1").strip().lower() in {"0", "false", "no"}:
            return
        try:
            _default_kb_search("plazo de desistimiento", 1)
        except Exception as exc:  # noqa: BLE001 - the tool will report it as an observation
            log.warning("knowledge base not available yet: %s: %s", type(exc).__name__, exc)

    # ---------------------------------------------------------------------- brains

    def _load_small(self) -> tuple:
        if self._small is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = BRAINS["base"]()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32
            ).to(device)
            self._small = (tokenizer, model.eval())
        return self._small

    def _load_thinking(self) -> tuple:
        if self._thinking is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = BRAINS["thinking"]()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32
            ).to(device)
            self._thinking = (tokenizer, model.eval())
        return self._thinking

    def _handles(self, brain: str) -> tuple:
        """``(tokenizer, model, context)``: the context disables the adapter for ``base``."""
        if brain == "thinking":
            tokenizer, model = self._load_thinking()
            return tokenizer, model, nullcontext()
        tokenizer, model = self._load_small()
        if brain == "rlm":
            if self._small_peft is None:
                path = rlm_adapter_path()
                if path is None:
                    raise RuntimeError(
                        "brain 'rlm' needs the phase-1 LoRA adapter: train it (rlm.train_grpo) "
                        "or set ARCA_RLM_ADAPTER to its folder (with adapter_config.json in it)."
                    )
                from peft import PeftModel

                self._small_peft = PeftModel.from_pretrained(model, path).eval()
            return tokenizer, self._small_peft, nullcontext()
        if self._small_peft is not None:  # base after rlm: the same weights, adapter off
            return tokenizer, self._small_peft, self._small_peft.disable_adapter()
        return tokenizer, model, nullcontext()

    def model_name(self, brain: str) -> str:
        if brain == "rlm":
            return f"{BRAINS['base']()}+{rlm_adapter_path() or 'no-adapter'}"
        return BRAINS[brain]()

    def generate(self, messages: list[dict], brain: str) -> str:
        """One greedy generation with the chosen brain (thinking on only for ``thinking``)."""
        import torch

        with self._lock:
            tokenizer, model, context = self._handles(brain)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=brain == "thinking",
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad(), context:
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS[brain],
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def count_tokens(self, text: str) -> int:
        """Tokens of ``text`` with the Qwen tokenizer if one is loaded, else a rough estimate
        (the Qwen3 family shares its tokenizer, so any loaded brain gives the same count)."""
        loaded = self._small or self._thinking
        if loaded is not None:
            return len(loaded[0].encode(text))
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------------ loop

    def system_prompt(self, max_steps: int) -> str:
        return (
            REACT_SYSTEM_PROMPT.format(
                max_steps=max_steps, tools_description=self.registry.render_for_prompt()
            )
            + AGENT_RULES
        )

    def _execute(self, call: ToolCall, seen: dict[str, int]) -> ToolCallRecord:
        """Run one tool call with every guard. Never raises."""
        if call.error is not None:
            return ToolCallRecord(
                name=call.name or "<invalid>",
                arguments={},
                ok=False,
                error=call.error + PARSE_ERROR_HINT,
            )
        key = _call_key(call.name, call.arguments)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=(
                    f"duplicate call: you already called {call.name} with these arguments "
                    "and got its result. Use that result, or call final_answer."
                ),
            )
        spec = self.registry.tools.get(call.name)
        if spec is not None and spec.requires_confirmation:
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
        timeout = self.kb_timeout_s if call.name == "search_knowledge_base" else self.tool_timeout_s
        try:
            return call_with_timeout(lambda: self.registry.call(call.name, call.arguments), timeout)
        except TimeoutError as exc:
            return ToolCallRecord(
                name=call.name,
                arguments=call.arguments,
                ok=False,
                error=f"timeout: {call.name} gave {exc}. Try other arguments or another tool.",
            )

    def _summarizer(self, brain: str) -> Callable[[list[dict]], str]:
        if self.summary_mode == "llm":

            def summarize(old: list[dict]) -> str:
                history = extractive_summary(old)
                prompt = SUMMARY_PROMPT.format(history=history)
                return self.generate([{"role": "user", "content": prompt}], brain).strip()

            return summarize
        return extractive_summary

    def run(self, task: str, max_steps: int = 10, brain: str = "rlm") -> AgentResponse:
        """The ReAct loop: think, act, observe, until ``final_answer`` or no steps left."""
        if brain not in BRAINS:
            raise ValueError(f"brain must be one of {sorted(BRAINS)}, not {brain!r}")
        memory = ConversationMemory(
            self.system_prompt(max_steps),
            self.count_tokens,
            token_budget=self.token_budget,
            pin_first=True,
            max_observation_chars=300,
        )
        memory.add("user", task)
        summarize = self._summarizer(brain)
        steps: list[AgentStep] = []
        seen: dict[str, int] = {}
        tokens = 0
        used_a_tool = False
        reminded = False
        model_name = self.model_name(brain)

        def finish(answer: str, ok: bool, n_steps: int) -> AgentResponse:
            self.steps = steps
            return AgentResponse(
                final_answer=answer,
                steps=steps,
                n_steps=n_steps,
                succeeded=ok,
                model=model_name,
                tokens_used=tokens,
            )

        for step_number in range(1, max_steps + 1):
            messages = memory.render()
            raw = self.generate(messages, brain)
            tokens += self.count_tokens("".join(m["content"] for m in messages))
            tokens += self.count_tokens(raw)
            raw = clean_model_output(raw)
            parsed = parse_tool_calls(raw)
            thought = _thought(parsed)
            valid = [c for c in parsed.tool_calls if c.error is None]
            first_is_broken = bool(parsed.tool_calls) and parsed.tool_calls[0].error is not None

            if not parsed.tool_calls:  # the model wrote prose instead of an action
                prose = _prose_answer(parsed)
                if prose and (used_a_tool or reminded):
                    steps.append(
                        AgentStep(
                            thought="(no final_answer call; the text is taken as the answer) "
                            + thought,
                            action=AgentAction(name="final_answer", arguments={"answer": prose}),
                            observation=prose,
                        )
                    )
                    return finish(prose, True, step_number)
                steps.append(AgentStep(thought=thought, observation=f"[reminder] {NUDGE}"))
                memory.add("assistant", raw)
                memory.add("user", NUDGE)
                reminded = True
                continue

            if first_is_broken or not valid:  # tell the model what is wrong, run nothing later
                record = self._execute(parsed.tool_calls[0], seen)
                observation = f"ERROR: {record.error}"
                steps.append(AgentStep(thought=thought, observation=clip(observation, 4000)))
                memory.add("assistant", raw)
                memory.add("tool", observation, name="error")
                memory.compress(summarize)
                continue

            call = valid[0]
            extra = (
                " Only your first tool call was executed; wait for results before the next."
                if len(valid) > 1
                else ""
            )
            action = AgentAction(name=call.name, arguments=call.arguments)

            if call.name == "final_answer":
                record = self.registry.call("final_answer", call.arguments)
                if record.ok:
                    answer = str(record.result)
                    steps.append(AgentStep(thought=thought, action=action, observation=answer))
                    return finish(answer, True, step_number)
            else:
                record = self._execute(call, seen)

            if record.ok:
                used_a_tool = True
                payload = format_tool_result(record.name, record.result)
            else:
                payload = f"ERROR: {record.error}"
            payload = clip(payload, self.max_observation_chars) + extra
            steps.append(AgentStep(thought=thought, action=action, observation=payload))
            memory.add("assistant", raw)
            memory.add("tool", payload, name=call.name)
            memory.compress(summarize)

            if seen.get(_call_key(call.name, call.arguments), 0) >= 3:
                log.warning("loop detected: %s called 3 times with the same arguments", call.name)
                return finish("", False, step_number)

        return finish("", False, max_steps)
