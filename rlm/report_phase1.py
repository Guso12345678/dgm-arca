"""Phase 1 report helpers: reward/length curves, pass@1 per problem type and a failure dump.

Three independent steps, all reading files the training and evaluation already wrote::

    # 1. curves from the GRPO log (or from a checkpoint's trainer_state.json)
    uv run python -m rlm.report_phase1 curves --log grpo.log

    # 2. pass@1 per type, base / SFT / GRPO, from reports/phase1_eval.json
    uv run python -m rlm.report_phase1 by-type --test rlm/data/test.jsonl

    # 3. failures of one model, with a rough reason, ready to be analysed by hand
    uv run python -m rlm.report_phase1 failures --model grpo --n 8
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TYPE_KEYS = ("type", "tipo", "category", "kind", "problem_type")
TYPE_PATTERNS = (
    ("tae", re.compile(r"\bTAE\b", re.I)),
    ("cuota", re.compile(r"\bcuota\b", re.I)),
    ("coste_total", re.compile(r"coste total|total (?:a pagar|pagado)|intereses totales", re.I)),
    ("ratio", re.compile(r"ratio|endeudamiento|DTI|esfuerzo", re.I)),
)


def parse_log(path: Path) -> list[dict]:
    """Every dict that TRL's logger printed with a training reward in it."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        start = line.find("{")
        if start < 0 or "'reward'" not in line:
            continue
        try:
            row = ast.literal_eval(line[start : line.rfind("}") + 1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(row, dict) and "reward" in row:
            rows.append(row)
    return rows


def parse_trainer_state(path: Path) -> list[dict]:
    state = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in state.get("log_history", []) if "reward" in r]


def column(rows: list[dict], *names: str) -> list[float | None]:
    return [next((r[n] for n in names if n in r), None) for r in rows]


def cmd_curves(args: argparse.Namespace) -> None:
    if args.state:
        rows = parse_trainer_state(Path(args.state))
    else:
        rows = parse_log(Path(args.log))
    if not rows:
        raise SystemExit("no reward rows found: check the path (log lines must contain 'reward')")
    steps = [r.get("step", i + 1) for i, r in enumerate(rows)]
    series = {
        "reward (total)": column(rows, "reward"),
        "format_reward": column(rows, "rewards/format_reward/mean"),
        "accuracy_reward": column(rows, "rewards/accuracy_reward/mean"),
        "domain_reward": column(rows, "rewards/domain_reward/mean"),
        "length (tokens)": column(rows, "completions/mean_length"),
        "reward_std": column(rows, "reward_std"),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1_grpo_history.json").write_text(
        json.dumps({"steps": steps, **series}, indent=2), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, names, title in (
        (axes[0], ["reward (total)"], "Recompensa total (media por paso)"),
        (axes[1], ["format_reward", "accuracy_reward", "domain_reward"], "Por recompensa"),
        (axes[2], ["length (tokens)"], "Longitud media de la completion"),
    ):
        for name in names:
            ys = series[name]
            if any(y is not None for y in ys):
                ax.plot(
                    [s for s, y in zip(steps, ys, strict=True) if y is not None],
                    [y for y in ys if y is not None],
                    label=name,
                )
        ax.set_title(title)
        ax.set_xlabel("paso GRPO")
        ax.grid(alpha=0.3)
        if len(names) > 1:
            ax.legend()
    fig.tight_layout()
    fig.savefig(out / "phase1_grpo_curves.png", dpi=150)
    first, last = (series["reward (total)"][0], series["reward (total)"][-1])
    print(f"{len(rows)} pasos · reward {first:.3f} -> {last:.3f}")
    print(f"-> {out / 'phase1_grpo_history.json'}, {out / 'phase1_grpo_curves.png'}")


def load_types(test_path: Path | None) -> dict[str, str]:
    """question -> type, from a field of the test file when there is one."""
    if not test_path or not test_path.exists():
        return {}
    types = {}
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = next((k for k in TYPE_KEYS if k in row), None)
        if key:
            types[row["question"]] = str(row[key])
    return types


def type_of(row: dict, known: dict[str, str]) -> str:
    if row["question"] in known:
        return known[row["question"]]
    for name, pattern in TYPE_PATTERNS:
        if pattern.search(row["question"]):
            return name
    return "otros"


def cmd_by_type(args: argparse.Namespace) -> None:
    results = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    known = load_types(Path(args.test) if args.test else None)
    table: dict[str, dict[str, tuple[int, int]]] = {}
    for model, data in results.items():
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in data["rows"]:
            c = counts[type_of(row, known)]
            c[0] += bool(row["is_correct"])
            c[1] += 1
        table[model] = {t: (c[0], c[1]) for t, c in counts.items()}
    types = sorted({t for per in table.values() for t in per})
    models = list(table)
    print(f"{'tipo':<14}" + "".join(f"{m:>16}" for m in models))
    for t in [*types, "TOTAL"]:
        cells = []
        for m in models:
            if t == "TOTAL":
                ok = sum(v[0] for v in table[m].values())
                n = sum(v[1] for v in table[m].values())
            else:
                ok, n = table[m].get(t, (0, 0))
            cells.append(f"{ok}/{n} = {ok / n:.3f}" if n else "-")
        print(f"{t:<14}" + "".join(f"{c:>16}" for c in cells))
    for m in models:
        rows = results[m]["rows"]
        valid = sum(r["has_valid_format"] for r in rows) / max(len(rows), 1)
        length = sum(r["n_tokens"] for r in rows) / max(len(rows), 1)
        print(f"{m}: formato válido {valid:.2f} · tokens medios {length:.0f}")


def reason(row: dict, max_new_tokens: int) -> str:
    """A rough first label for a failure; the real explanation is yours to write."""
    if not row["has_valid_format"]:
        return "truncado" if row["n_tokens"] >= max_new_tokens - 2 else "formato inválido"
    if row["predicted"] in (None, ""):
        return "sin respuesta"
    try:
        pred = float(str(row["predicted"]).replace(",", "."))
        gold = float(str(row["expected"]).replace(",", "."))
    except ValueError:
        return "respuesta no numérica"
    if gold and abs(pred - gold) / abs(gold) < 0.05:
        return "casi correcto (redondeo/unidades)"
    ratio = pred / gold if gold else 0
    if abs(ratio - 100) < 1e-6 or abs(ratio - 0.01) < 1e-9:
        return "factor 100 (% vs fracción)"
    return "cálculo o planteamiento erróneo"


def cmd_failures(args: argparse.Namespace) -> None:
    results = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    known = load_types(Path(args.test) if args.test else None)
    rows = [r for r in results[args.model]["rows"] if not r["is_correct"]]
    labels = Counter(reason(r, args.max_new_tokens) for r in rows)
    print(f"{args.model}: {len(rows)} fallos · {dict(labels)}\n")
    # one per type first, so the analysis covers the different kinds of error
    seen, picked = set(), []
    for r in sorted(rows, key=lambda r: type_of(r, known)):
        key = (type_of(r, known), reason(r, args.max_new_tokens))
        if key not in seen:
            seen.add(key)
            picked.append(r)
    picked += [r for r in rows if r not in picked]
    lines = []
    for i, r in enumerate(picked[: args.n], 1):
        lines += [
            f"## Fallo {i} · {type_of(r, known)} · {reason(r, args.max_new_tokens)}",
            f"**Pregunta:** {r['question']}",
            f"**Esperado:** {r['expected']}  ·  **Predicho:** {r['predicted']}  "
            f"·  tokens: {r['n_tokens']}  ·  formato válido: {r['has_valid_format']}",
            "**Salida (recortada):**",
            "```",
            r["raw"][: args.clip] + (" [...]" if len(r["raw"]) > args.clip else ""),
            "```",
            "",
        ]
    text = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text[:3000])
    print(f"\n-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("curves")
    p.add_argument("--log", default="grpo.log")
    p.add_argument("--state", default=None, help="trainer_state.json instead of the log")
    p.add_argument("--out-dir", default="reports")
    p.set_defaults(fn=cmd_curves)

    for name, fn in (("by-type", cmd_by_type), ("failures", cmd_failures)):
        p = sub.add_parser(name)
        p.add_argument("--eval", default="reports/phase1_eval.json")
        p.add_argument("--test", default="rlm/data/test.jsonl")
        if name == "failures":
            p.add_argument("--model", default="grpo")
            p.add_argument("--n", type=int, default=8)
            p.add_argument("--clip", type=int, default=900)
            p.add_argument("--max-new-tokens", type=int, default=1024)
            p.add_argument("--out", default="reports/phase1_failures.md")
        p.set_defaults(fn=fn)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
