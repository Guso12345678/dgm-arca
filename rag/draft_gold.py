"""Draft gold-set candidates with a model, then review and promote the good ones.

Writing fifty questions from scratch is the slowest part of phase 3. This tool speeds up the
*drafting*, never the *judging*: a model reads a sample of chunks and proposes a question and
the verbatim passage that answers it; the code rejects every proposal whose evidence is not
literally in the chunk; and a person decides which ones enter ``rag/gold/gold.jsonl``.

    uv run python -m rag.draft_gold --n 60                  # writes draft.jsonl + draft_review.md
    # read rag/gold/draft_review.md, note the numbers of the good questions
    uv run python -m rag.draft_gold --promote 1,4,7-12      # appends them to gold.jsonl

Things to keep in mind (and to write in the report):

* Questions derived from a chunk tend to reuse its words, which favours BM25 over dense
  retrieval. The prompt asks for paraphrases and every draft carries its ``lexical_overlap``
  (share of the question's content words that appear in the passage); the promote step prints
  the mean overlap of the final gold set so you can say how lexical it is.
* A model never writes the hard questions (two passages, a deadline spread over two articles,
  a negation). Write at least a third of the gold set yourselves.
* Say in the report that part of the gold set was drafted by a model and reviewed by hand.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from rag.evaluate import load_gold, normalise
from rag.retriever import tokenize

GOLD_DIR = Path(__file__).parent / "gold"
DEFAULT_MODEL = "Qwen/Qwen3-4B"
MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 15, 300
MIN_EVIDENCE_CHARS, MAX_EVIDENCE_CHARS = 20, 200
HIGH_OVERLAP = 0.8

PROMPT = """Eres un analista de riesgo de crédito que consulta normativa sobre crédito al consumo.
Lee el pasaje y escribe UNA pregunta que un analista podría hacer y que el pasaje responde.

Reglas:
- La pregunta debe poder entenderse sin ver el pasaje (no digas "según el pasaje" ni "el texto").
- Parafrasea: evita copiar las palabras clave del pasaje; usa sinónimos y otra estructura.
- Pregunta por un hecho concreto (plazo, importe, obligación, definición, excepción).
- "evidence" es una cita LITERAL, copiada carácter por carácter del pasaje, de entre 40 y 150
  caracteres, que contiene la respuesta.
- Responde SOLO con un JSON: {{"question": "...", "evidence": "..."}}

Pasaje:
{passage}
"""


def lexical_overlap(question: str, passage: str) -> float:
    """Share of the question's content words (4+ letters) that also appear in the passage."""
    words = {w for w in tokenize(question) if len(w) >= 4}
    if not words:
        return 0.0
    passage_words = set(tokenize(passage))
    return len(words & passage_words) / len(words)


def parse_proposal(text: str) -> dict | None:
    """First JSON object in a model reply, tolerating <think> blocks and ``` fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not all(
        isinstance(data.get(k), str) for k in ("question", "evidence")
    ):
        return None
    return {"question": data["question"].strip(), "evidence": data["evidence"].strip()}


def validate_proposal(proposal: dict, passage: str) -> str | None:
    """None when the proposal is acceptable, otherwise the reason it is rejected."""
    question, evidence = proposal["question"], proposal["evidence"]
    if not (MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS) or "?" not in question:
        return "bad question"
    if re.search(r"según (el|este) (pasaje|texto|fragmento)", question, flags=re.IGNORECASE):
        return "refers to the passage"
    if not (MIN_EVIDENCE_CHARS <= len(evidence) <= MAX_EVIDENCE_CHARS):
        return "evidence length"
    if normalise(evidence) not in normalise(passage):
        return "evidence is not literal"
    return None


def usable_chunk(text: str, min_chars: int) -> bool:
    """Prose, not tables, lists of references or numbers."""
    letters = sum(c.isalpha() for c in text)
    return len(text) >= min_chars and letters / max(len(text), 1) >= 0.75


def sample_chunks(chunks: list, n: int, seed: int, min_chars: int = 450) -> list:
    """Round-robin over documents so a long document does not take every slot."""
    by_source: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        if usable_chunk(chunk.text, min_chars):
            by_source[chunk.source].append(chunk)
    rng = random.Random(seed)
    for group in by_source.values():
        rng.shuffle(group)
    picked: list = []
    while len(picked) < n and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source] and len(picked) < n:
                picked.append(by_source[source].pop())
    return picked


def draft(
    chunks: list, generate_fn: Callable[[str], str], n: int, seed: int = 0
) -> tuple[list, dict]:
    """Propose, validate and de-duplicate. Returns ``(rows, rejection_counts)``."""
    rows, rejected, seen = [], defaultdict(int), set()
    for chunk in sample_chunks(chunks, n, seed):
        proposal = parse_proposal(generate_fn(PROMPT.format(passage=chunk.text)))
        if proposal is None:
            rejected["unparseable"] += 1
            continue
        reason = validate_proposal(proposal, chunk.text)
        if reason is None and normalise(proposal["question"]) in seen:
            reason = "duplicate"
        if reason:
            rejected[reason] += 1
            continue
        seen.add(normalise(proposal["question"]))
        overlap = round(lexical_overlap(proposal["question"], chunk.text), 2)
        note = "DRAFT (model-proposed, review)"
        if overlap >= HIGH_OVERLAP:
            note += "; high lexical overlap: favours BM25"
        rows.append(
            {
                "question": proposal["question"],
                "evidence": proposal["evidence"],
                "source": chunk.source,
                "notes": note,
                "lexical_overlap": overlap,
                "_context": chunk.text,
            }
        )
    return rows, dict(rejected)


def write_review(rows: list, path: Path) -> None:
    lines = ["# Gold-set drafts: read, then `--promote` the numbers you accept", ""]
    for i, row in enumerate(rows, start=1):
        lines += [
            f"## {i}. {row['question']}",
            f"- source: `{row['source']}` · lexical overlap: {row['lexical_overlap']}",
            f"- evidence: «{row['evidence']}»",
            "",
            f"> {row['_context'][:900].replace(chr(10), ' ')}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_selection(spec: str, total: int) -> list[int]:
    """'1,4,7-9' -> [1, 4, 7, 8, 9] (1-based), validated against ``total``."""
    chosen: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            chosen.extend(range(low, high + 1))
        else:
            chosen.append(int(part))
    bad = [i for i in chosen if not 1 <= i <= total]
    if bad:
        raise ValueError(f"numbers out of range 1..{total}: {bad}")
    return list(dict.fromkeys(chosen))


def promote(rows: list, selection: list[int], gold_path: Path) -> dict:
    """Append the selected drafts to the gold file (without internal fields); skip repeats."""
    existing = load_gold(gold_path) if gold_path.exists() else []
    known = {normalise(item["question"]) for item in existing}
    added = []
    for i in selection:
        row = rows[i - 1]
        if normalise(row["question"]) in known:
            continue
        added.append({k: v for k, v in row.items() if k != "_context"})
    if added:
        with gold_path.open("a", encoding="utf-8") as handle:
            for item in added:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    final = existing + added
    overlaps = [i["lexical_overlap"] for i in final if "lexical_overlap" in i]
    return {
        "added": len(added),
        "skipped_repeats": len(selection) - len(added),
        "gold_total": len(final),
        "mean_lexical_overlap_of_drafted": round(sum(overlaps) / len(overlaps), 2)
        if overlaps
        else None,
    }


def load_generator(model_id: str) -> Callable[[str], str]:
    from rag.generate import load_chat_generator

    chat = load_chat_generator(model_id, 300)
    return lambda prompt: chat([{"role": "user", "content": prompt}])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=60, help="chunks to sample (drafts <= n)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--promote", metavar="NUMBERS", help="e.g. 1,4,7-12")
    parser.add_argument("--gold", default=str(GOLD_DIR / "gold.jsonl"))
    args = parser.parse_args()

    draft_path = GOLD_DIR / "draft.jsonl"
    if args.promote:
        rows = [json.loads(line) for line in draft_path.read_text(encoding="utf-8").splitlines()]
        report = promote(rows, parse_selection(args.promote, len(rows)), Path(args.gold))
        print(json.dumps(report, indent=2))
        return

    from rag.chunking import recursive_chunks
    from rag.ingest import load_corpus

    chunks = recursive_chunks(load_corpus(), args.chunk_size, 0)
    rows, rejected = draft(chunks, load_generator(args.model), args.n, args.seed)
    GOLD_DIR.mkdir(exist_ok=True)
    draft_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    write_review(rows, GOLD_DIR / "draft_review.md")
    print(f"{len(rows)} drafts from {args.n} chunks; rejected: {rejected}")
    print(f"read {GOLD_DIR / 'draft_review.md'} and promote with --promote 1,4,7-12")


if __name__ == "__main__":
    main()
