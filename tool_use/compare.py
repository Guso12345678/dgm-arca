"""Phase 2 comparison: the same model, with and without tools, on the phase-1 credit problems.

For every problem of the phase-1 test set the model (Qwen3-1.7B by default) answers twice:

* **without tools**: just the question, as a plain chat. It has to do the arithmetic itself.
* **with tools**: the full tool loop (``ToolLoop.run``), so it can call ``loan_metrics``.

The question is the same text in both conditions (it already carries the "two decimals" rule
of the phase-1 dataset). The metric is the same too: an answer is correct when any number in
it matches the expected value to the cent. That is deliberately lenient about *format* (the
tool loop answers in free text, not in ``<answer>`` tags) and strict about the *value*.
Numbers inside ``<think>`` blocks are ignored.

Run::

    uv run python -m tool_use.compare --data rlm/data/rlm_data_test.jsonl --per-type 10 \\
        --out reports/phase2_compare.json

The expected picture, and the reason the practice asks for this comparison: without tools
the model fails on anything that needs powers or iteration (instalment, APR) and does fine
on plain arithmetic (total cost, debt ratio, once the instalment is given); with tools the
gap closes, as long as the model calls the tool with the right arguments.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
NUMBER_PATTERN = re.compile(r"\d[\d.,]*")
TOLERANCE = 0.005 + 1e-9  # to the cent
THOUSANDS_GROUPS = re.compile(r"^\d{1,3}([.,]\d{3})+$")


def parse_number(token: str) -> float | None:
    """Parse '378.81', '378,81', '1,948.25' and '1.948,25'; None when it is not a number.

    Rules: if both separators appear, the last one is the decimal mark. A single separator
    is a decimal mark, except when the token is a repeated thousands grouping (1,948,000 or
    12.345.678) or a comma followed by exactly three digits (1,948).
    """
    token = token.strip(".,")
    if not token:
        return None
    if "." in token and "," in token:
        decimal = "." if token.rfind(".") > token.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        token = token.replace(thousands, "").replace(decimal, ".")
    elif "," in token:
        head, _, tail = token.rpartition(",")
        if THOUSANDS_GROUPS.match(token) and (len(tail) == 3):
            token = token.replace(",", "")
        else:
            token = head.replace(",", "") + "." + tail
    elif token.count(".") > 1 and THOUSANDS_GROUPS.match(token):
        token = token.replace(".", "")
    try:
        return float(token)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    """All numbers of a model answer, ignoring ``<think>`` blocks and markdown emphasis."""
    clean = THINK_PATTERN.sub("", text).replace("*", "")
    values = (parse_number(token) for token in NUMBER_PATTERN.findall(clean))
    return [value for value in values if value is not None]


def is_correct(text: str, expected: str) -> bool:
    """True when some number of the answer equals the expected value to the cent."""
    target = float(expected)
    return any(abs(value - target) <= TOLERANCE for value in numbers_in(text))


def load_problems(path: str | Path, per_type: int | None, seed: int = 0) -> list[dict[str, Any]]:
    """Read the phase-1 JSONL (fields ``tipo``, ``question``, ``answer``); sample per type."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if per_type is None:
        return rows
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["tipo"]].append(row)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for tipo in sorted(by_type):
        group = by_type[tipo]
        picked.extend(rng.sample(group, min(per_type, len(group))))
    return picked


def evaluate(loop, problems: list[dict[str, Any]], max_turns: int = 5) -> list[dict[str, Any]]:
    """Answer every problem without and with tools. ``loop`` is a loaded ``ToolLoop``."""
    rows = []
    for index, problem in enumerate(problems, start=1):
        question, expected = problem["question"], problem["answer"]

        started = time.perf_counter()
        plain = loop.generate([{"role": "user", "content": question}])
        plain_seconds = time.perf_counter() - started

        started = time.perf_counter()
        response = loop.run(question, max_turns=max_turns)
        tools_seconds = time.perf_counter() - started

        row = {
            "id": problem.get("id"),
            "tipo": problem["tipo"],
            "expected": expected,
            "without_tools": {
                "answer": plain,
                "correct": is_correct(plain, expected),
                "seconds": round(plain_seconds, 2),
            },
            "with_tools": {
                "answer": response.answer,
                "correct": is_correct(response.answer, expected),
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments, "ok": c.ok}
                    for c in response.tool_calls
                ],
                "turns": response.turns,
                "seconds": round(tools_seconds, 2),
            },
        }
        rows.append(row)
        print(
            f"[{index}/{len(problems)}] {row['tipo']:<22} "
            f"sin={'✓' if row['without_tools']['correct'] else '✗'} "
            f"con={'✓' if row['with_tools']['correct'] else '✗'}",
            flush=True,
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Accuracy with / without tools per problem type and overall."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["tipo"]].append(row)
        groups["TOTAL"].append(row)

    summary = {}
    for name, group in groups.items():
        n = len(group)
        used_tool = sum(
            any(c["name"] == "loan_metrics" and c["ok"] for c in r["with_tools"]["tool_calls"])
            for r in group
        )
        summary[name] = {
            "n": n,
            "without_tools": sum(r["without_tools"]["correct"] for r in group) / n,
            "with_tools": sum(r["with_tools"]["correct"] for r in group) / n,
            "used_loan_metrics": used_tool / n,
        }
    return summary


def print_table(summary: dict[str, dict[str, float]]) -> None:
    print(f"\n{'tipo':<22}{'n':>4}{'sin tools':>12}{'con tools':>12}{'usa loan_metrics':>20}")
    for name in sorted(summary, key=lambda k: (k == "TOTAL", k)):
        s = summary[name]
        print(
            f"{name:<22}{s['n']:>4}{s['without_tools']:>12.2f}{s['with_tools']:>12.2f}"
            f"{s['used_loan_metrics']:>20.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="rlm/data/rlm_data_test.jsonl")
    parser.add_argument("--per-type", type=int, default=10, help="problems per type (0 = all)")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports/phase2_compare.json")
    args = parser.parse_args()

    from tool_use.executor import build_tool_loop

    problems = load_problems(args.data, args.per_type or None, args.seed)
    rows = evaluate(build_tool_loop(), problems, args.max_turns)
    summary = summarise(rows)
    print_table(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(f"\ndetails -> {out}")


if __name__ == "__main__":
    main()