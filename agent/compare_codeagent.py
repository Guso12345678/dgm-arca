"""JSON agent against a code agent: the same brains, the same tools, the same benchmark.

Our agent (``agent/react_agent.py``) acts by emitting one JSON tool call per step. The
``CodeAgent`` of smolagents acts by writing a Python snippet that calls the tools and can chain
them in one step (the CodeAct idea). Both use the *same* model through the same
``ReActAgent.generate`` (greedy), the same tools from the phase-2 registry (plus the knowledge
base) and the same automatic checks, so the only thing that changes is the action format.

    ARCA_AGENT_FIXTURE=bench ARCA_REPORTS_DIR=/tmp/reports \\
        uv run python -m agent.compare_codeagent --bench agent/bench/tasks.jsonl \\
        --brains thinking --limit 8

What is measured per task: success (the automatic check), steps, tokens and whether the
required tools appear in the code the model wrote. Read the table with care: a 0.6B model and
a 1.7B one write very different code, and a handful of tasks is a handful of tasks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from agent.evaluate import check_answer
from agent.react_agent import ReActAgent, final_answer, freeze_euribor, search_knowledge_base

JSON_TO_SMOLAGENTS = {
    "string": "string",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "object": "object",
    "array": "array",
}
ROLE_MAP = {"tool-call": "assistant", "tool-response": "user"}


def _json_type(prop: dict) -> tuple[str, bool]:
    """``(smolagents type, nullable)`` for one JSON-Schema property."""
    if "anyOf" in prop:
        kinds = [p for p in prop["anyOf"] if p.get("type") != "null"]
        nullable = len(kinds) != len(prop["anyOf"])
        return _json_type(kinds[0])[0] if kinds else "any", nullable
    return JSON_TO_SMOLAGENTS.get(prop.get("type", "string"), "any"), False


def to_smolagents_tool(registry_tool, registry):
    """Wrap one phase-2 tool as a smolagents ``Tool`` with the same name and description."""
    from smolagents import Tool

    schema = registry_tool.definition["function"]["parameters"]
    required = set(schema.get("required", []))
    inputs: dict[str, dict[str, Any]] = {}
    for name, prop in schema.get("properties", {}).items():
        kind, nullable = _json_type(prop)
        entry: dict[str, Any] = {"type": kind, "description": prop.get("description") or name}
        if nullable or name not in required:
            entry["nullable"] = True
        inputs[name] = entry

    ordered = sorted(inputs, key=lambda n: n not in required)  # required parameters first
    signature = ", ".join(n if n in required else f"{n}=None" for n in ordered)
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102 - builds a method with the tool's own argument names, nothing else
        f"def forward(self, {signature}):\n"
        f"    return _run({{k: v for k, v in dict({', '.join(f'{n}={n}' for n in ordered)}).items()"
        " if v is not None})",
        {"_run": lambda arguments: _unwrap(registry.call(registry_tool.name, arguments))},
        namespace,
    )
    return type(
        f"{registry_tool.name}_tool",
        (Tool,),
        {
            "name": registry_tool.name,
            "description": registry_tool.description,
            "inputs": inputs,
            "output_type": "any",
            "forward": namespace["forward"],
        },
    )()


def _unwrap(record) -> Any:
    """A failed call raises, so the code agent sees the error in its next observation."""
    if not record.ok:
        raise RuntimeError(record.error)
    return record.result


def build_model(agent: ReActAgent, brain: str):
    """A smolagents ``Model`` backed by the same generation function as our JSON agent."""
    from smolagents.models import ChatMessage, Model
    from smolagents.monitoring import TokenUsage

    class BrainModel(Model):
        def generate(self, messages, stop_sequences=None, **kwargs):
            plain = []
            for message in messages:
                role = getattr(message.role, "value", message.role)
                content = message.content
                if isinstance(content, list):
                    content = "\n".join(part.get("text", "") for part in content)
                plain.append({"role": ROLE_MAP.get(role, role), "content": content or ""})
            text = agent.generate(plain, brain)
            for stop in stop_sequences or []:  # smolagents expects the model to stop here
                if stop in text:
                    text = text.split(stop)[0]
            usage = TokenUsage(
                input_tokens=agent.count_tokens("".join(m["content"] for m in plain)),
                output_tokens=agent.count_tokens(text),
            )
            return ChatMessage(role="assistant", content=text, token_usage=usage)

    return BrainModel(model_id=agent.model_name(brain))


def run_code_agent(agent: ReActAgent, brain: str, task: dict, max_steps: int) -> dict:
    from smolagents import CodeAgent

    tools = [
        to_smolagents_tool(t, agent.registry)
        for name, t in agent.registry.tools.items()
        if name != "final_answer"  # CodeAgent brings its own final_answer
    ]
    code_agent = CodeAgent(
        tools=tools, model=build_model(agent, brain), max_steps=max_steps, verbosity_level=0
    )
    try:
        answer = str(code_agent.run(task["task"]))
        error = ""
    except Exception as exc:  # noqa: BLE001 - one bad task must not stop the comparison
        answer, error = "", f"{type(exc).__name__}: {exc}"
    code = " ".join(
        str(call.arguments)
        for step in code_agent.memory.steps
        for call in (getattr(step, "tool_calls", None) or [])
    )
    steps = [s for s in code_agent.memory.steps if type(s).__name__ == "ActionStep"]
    used = {n for n in agent.registry.tools if re.search(rf"\b{n}\s*\(", code)}
    usage = code_agent.monitor.get_total_token_counts()
    return {
        **task,
        "final_answer": answer,
        "error": error,
        "n_steps": len(steps),
        "tokens_used": usage.input_tokens + usage.output_tokens,
        "correct": bool(answer) and check_answer(task, answer),
        "required_tools_used": set(task.get("required_tools", [])) <= used,
    }


def summarise(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "success_rate": sum(r["correct"] for r in rows) / n,
        "required_tools_rate": sum(r["required_tools_used"] for r in rows) / n,
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
        "n_tasks": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/tasks.jsonl")
    parser.add_argument("--brains", nargs="+", default=["thinking"])
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    parser.add_argument("--out", default="reports/phase4_codeagent.json")
    args = parser.parse_args()

    tasks = [
        json.loads(line)
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    agent = ReActAgent()
    agent.registry.register(final_answer, search_knowledge_base)
    freeze_euribor(agent.registry)
    agent.load()

    results: dict[str, dict] = {}
    for brain in args.brains:
        json_rows, code_rows = [], []
        for task in tasks:
            response = agent.run(task["task"], max_steps=args.max_steps, brain=brain)
            used = {s.action.name for s in response.steps if s.action}
            json_rows.append(
                {
                    **task,
                    "final_answer": response.final_answer,
                    "n_steps": response.n_steps,
                    "tokens_used": response.tokens_used,
                    "correct": response.succeeded and check_answer(task, response.final_answer),
                    "required_tools_used": set(task.get("required_tools", [])) <= used,
                }
            )
            code_rows.append(run_code_agent(agent, brain, task, args.max_steps))
        results[brain] = {
            "json_agent": {**summarise(json_rows), "rows": json_rows},
            "code_agent": {**summarise(code_rows), "rows": code_rows},
        }
        for kind in ("json_agent", "code_agent"):
            s = results[brain][kind]
            print(
                f"{brain:>9} {kind:>10}: success {s['success_rate']:.2f} "
                f"· required tools {s['required_tools_rate']:.2f} "
                f"· steps {s['mean_steps']:.1f} · tokens {s['mean_tokens']:.0f}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))


if __name__ == "__main__":
    main()
