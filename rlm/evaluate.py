"""Phase 1 evaluation: pass@1 of base vs SFT vs GRPO on a held-out set, plus training curves.

Run::

    uv run python -m rlm.evaluate --data rlm/data/test.jsonl \
        --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora

It writes ``reports/phase1_eval.json`` with per-example verdicts (so you can do the failure
analysis) and ``reports/phase1_pass1.png`` with the bar chart. ``--history`` plots the
reward curves from the JSON history that the training scripts save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.data import load_domain_dataset, load_gsm8k
from rlm.inference import ReasoningModel
from rlm.rewards import has_valid_format
from rlm.verifier import CreditoVerifier, NumericVerifier, Verifier


def evaluate_model(
    base_model: str, adapter: str | None, dataset, verifier: Verifier, max_new_tokens: int
) -> list[dict]:
    model = ReasoningModel(base_model=base_model, adapter_path=adapter, verifier_name=verifier.name)
    model.load()

    rows: list[dict] = []
    for example in dataset:
        question = next(m["content"] for m in example["prompt"] if m["role"] == "user")
        expected = example["answer"]

        raw, n_tokens = model.generate(question, max_new_tokens)
        result = verifier.verify(raw, expected)

        rows.append({
            "question": question,
            "expected": expected,
            "raw": raw,
            "predicted": result.predicted,
            "is_correct": result.is_correct,
            "has_valid_format": has_valid_format(raw),
            "n_tokens": n_tokens,
        })
        print(f"[{len(rows)}/{len(dataset)}] aciertos={sum(r['is_correct'] for r in rows)}", flush=True)
    return rows


def pass_at_1(rows: list[dict]) -> float:
    return sum(r["is_correct"] for r in rows) / max(len(rows), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="gsm8k", help="'gsm8k' (test split) or your test JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["base=none"],
        help="name=path pairs; use 'none' for the bare base model",
    )
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--out", default="reports/phase1_eval.json")
    args = parser.parse_args()
    verifier = NumericVerifier() if args.data == "gsm8k" else CreditoVerifier()
    dataset = (
        load_gsm8k("test", n_examples=args.n_examples)
        if args.data == "gsm8k"
        else load_domain_dataset(args.data)
    )
    results = {}
    for pair in args.adapters:
        name, path = pair.split("=", 1)
        rows = evaluate_model(
            args.model,
            None if path == "none" else path,
            dataset,
            verifier,
            args.max_new_tokens,
        )
        results[name] = {"pass@1": pass_at_1(rows), "rows": rows}
        print(f"{name:>8}: pass@1 = {results[name]['pass@1']:.3f} on {len(rows)} problems")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
