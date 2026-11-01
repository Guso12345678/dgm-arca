"""Phase 1, step 1b: generate reasoning traces with a teacher model and keep the verified ones.

This is what Sky-T1, OpenThoughts and DeepSeek's cold start have in common: a strong
model writes solutions with visible reasoning, a verifier throws away the wrong ones,
and what survives becomes SFT data. Here the teacher is any model that can think in the
``<think>…</think><answer>…</answer>`` format (Qwen3 in thinking mode works well; a
DeepSeek-R1 distilled model too).

Run::

    uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-4B \
        --samples 4 --output rlm/data/sft_traces.jsonl

Output: one JSON line per generated trace with ``question``, ``answer``, ``trace``,
``verified`` and ``teacher``. Report in EXPERIMENTS.md the acceptance rate: it is your
first measurement of how hard your domain is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlm.data import load_domain_dataset
from rlm.rewards import has_valid_format, thinking_length
from rlm.verifier import CreditoVerifier, Verifier

MIN_THINKING_TOKENS = 15 # Actua como umbral de filtro de "correcto por suerte"
def _load_teacher(teacher: str):
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        teacher,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model

def _generate_batch(
    tokenizer, model, prompts: list[list[dict]], samples: int, max_new_tokens: int
) -> list[list[str]]:
    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=samples,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    completions = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
    grouped = [completions[i : i + samples] for i in range(0, len(completions), samples)]
    return grouped

#Para corregir el formato de la traza del profesor, si termina en \boxed{...} pero no usa <answer>
def _maybe_canonicalize(trace: str) -> str:
    if "<answer>" in trace:
        return trace
    if "\\boxed{" in trace:
        boxed_start = trace.rfind("\\boxed{")
        boxed_end = trace.find("}", boxed_start)
        if boxed_end != -1:
            value = trace[boxed_start + len("\\boxed{") : boxed_end]
            think_part = trace[:boxed_start].strip()
            if "<think>" not in think_part:
                think_part = f"<think>{think_part}</think>"
            return f"{think_part}<answer>{value}</answer>"
    return trace

def generate_traces(
    dataset, teacher: str, samples: int, max_new_tokens: int, verifier: Verifier,batch_size: int,
) -> list[dict]:
    """Tu turno: for each problem, sample ``samples`` completions from the teacher and verify them.

    Suggested steps:

    1. Load tokenizer and model (bf16 on GPU). Batch the prompts: generation dominates the cost.
    2. For each problem, ``generate`` with ``num_return_sequences=samples``, ``do_sample=True``.
    3. Decode, run ``verifier.verify(trace, answer)``, and store every trace with its verdict.
    4. Optional but recommended: if the teacher omits the ``<answer>`` tag but ends with
       ``\\boxed{...}``, rewrite the trace into the canonical format before saving.

    Watch out for traces that are correct by luck with nonsense reasoning: a second pass
    with an LLM judge, or a minimum-length filter, is a cheap way to catch some of them.
    """
    tokenizer, model = _load_teacher(teacher)
 
    questions = dataset["prompt"]       
    answers = dataset["answer"]         
    raw_questions = [
        next(m["content"] for m in msgs if m["role"] == "user") for msgs in questions
    ]
    results: list[dict] = []
 
    for start in range(0, len(questions), batch_size):
        batch_prompts = questions[start : start + batch_size]
        batch_answers = answers[start : start + batch_size]
        batch_questions_text = raw_questions[start : start + batch_size]
 
        grouped_completions = _generate_batch(tokenizer, model, batch_prompts, samples, max_new_tokens)
 
        for question_text, expected, completions in zip(
            batch_questions_text, batch_answers, grouped_completions, strict=True
        ):
            for raw_trace in completions:
                trace = _maybe_canonicalize(raw_trace.strip())
                verification = verifier.verify(trace, expected)
 
                verified = verification.is_correct
                if verified and has_valid_format(trace):
                    if thinking_length(trace) < MIN_THINKING_TOKENS:
                        verified = False
 
                results.append(
                    {
                        "question": question_text,
                        "answer": expected,
                        "trace": trace,
                        "verified": verified,
                        "teacher": teacher,
                    }
                )
        kept_so_far = sum(1 for r in results if r["verified"])
        print(
            f"[{min(start + batch_size, len(questions))}/{len(questions)}] "
            f"{kept_so_far}/{len(results)} trazas verificadas",
            flush=True,
        )
 
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="domain JSONL with question / answer")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4, help="traces per problem")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N problems")
    parser.add_argument("--batch-size", type=int, default=4, help="problems per generation batch")
    args = parser.parse_args()
 
    dataset = load_domain_dataset(args.data)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    traces = generate_traces(
        dataset, args.teacher, args.samples, args.max_new_tokens, CreditoVerifier(),batch_size=args.batch_size  
    )   
 
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept = sum(1 for t in traces if t["verified"])
    print(
        f"{kept}/{len(traces)} traces verified ({100 * kept / max(len(traces), 1):.1f}%) -> {out}"
    )
 
 
if __name__ == "__main__":
    main()