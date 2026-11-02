"""Phase 4 evaluation: success rate, steps and tokens on multi-tool tasks, per brain.

Each line of the benchmark (``agent/bench/example_tasks.jsonl``) has a task, the tools that
must be involved, and a way to check the final answer automatically (a number, a substring,
a regex). Twenty tasks that need at least two tools is the minimum. Run every brain::

    uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base rlm thinking

and put the resulting table in your report next to the JSON-vs-code-agent comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def check_answer(task: dict, final_answer: str) -> bool:
    """Automatic check of the final answer, according to the task's ``check`` field."""
    check = task.get("check", {})
    kind, value = check.get("type"), check.get("value")
    if kind == "contains":
        return str(value).lower() in final_answer.lower()
    if kind == "regex":
        return re.search(value, final_answer) is not None
    if kind == "number":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", final_answer.replace(",", ""))
        return any(abs(float(n) - float(value)) <= check.get("tolerance", 0) for n in numbers)
    return False


def tools_used(steps: list[dict]) -> set[str]:
    return {s["action"]["name"] for s in steps if s.get("action")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bench", default="agent/bench/example_tasks.jsonl")
    parser.add_argument("--brains", nargs="+", default=["rlm"])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out", default="reports/phase4_eval.json")
    args = parser.parse_args()

    from agent.react_agent import ReActAgent

    tasks = [
        json.loads(line)
        for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    agent = ReActAgent.from_env()
    results = {}
    for brain in args.brains:
        rows = []
        for task in tasks:
            response = agent.run(task["task"], max_steps=args.max_steps, brain=brain).model_dump()
            ok = response["succeeded"] and check_answer(task, response["final_answer"])
            used = tools_used(response["steps"])
            rows.append(
                {
                    **task,
                    **response,
                    "correct": ok,
                    "tools_used": sorted(used),
                    "required_tools_used": set(task.get("required_tools", [])) <= used,
                }
            )
        n = max(len(rows), 1)
        results[brain] = {
            "success_rate": sum(r["correct"] for r in rows) / n,
            "required_tools_rate": sum(r["required_tools_used"] for r in rows) / n,
            "mean_steps": sum(r["n_steps"] for r in rows) / n,
            "mean_tokens": sum(r["tokens_used"] for r in rows) / n,
            "rows": rows,
        }
        print(
            f"{brain:>9}: success {results[brain]['success_rate']:.2f} "
            f"· required tools {results[brain]['required_tools_rate']:.2f} "
            f"· steps {results[brain]['mean_steps']:.1f} "
            f"· tokens {results[brain]['mean_tokens']:.0f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=list))


if __name__ == "__main__":
    main()
