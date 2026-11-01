"""Build gold entries from (question, keywords) specs; the evidence is found in the corpus.

Writing the verbatim ``evidence`` by hand is the slow and error-prone part of a gold set. Here a
person (or a model) only writes the *question* and a few words that the answering sentence must
contain; the tool looks for that sentence in the ingested corpus and copies it literally, so the
evidence matches by construction. Judging stays human: every match is printed and must be read.

    uv run python -m rag.build_gold --spec rag/gold/spec.jsonl            # dry run: prints matches
    uv run python -m rag.build_gold --spec rag/gold/spec.jsonl --write    # appends to gold.jsonl

Spec line: {"question": "...", "source": "ley", "all": ["catorce", "desistir"]}
``source`` is a substring of the file name; ``all`` are words that must all appear in the sentence
(case and accents matter little, see ``rag.evaluate.normalise``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag.chunking import split_sentences
from rag.evaluate import load_gold, normalise
from rag.ingest import CORPUS_DIR, Document, load_corpus

GOLD_PATH = Path(__file__).parent / "gold" / "gold.jsonl"
MIN_EVIDENCE_CHARS, MAX_EVIDENCE_CHARS = 20, 200


def clip_evidence(sentence: str, keywords: list[str]) -> str:
    """Return a verbatim piece of the sentence of at most MAX chars, centred on a keyword."""
    sentence = " ".join(sentence.split())
    if len(sentence) <= MAX_EVIDENCE_CHARS:
        return sentence
    lowered = normalise(sentence)
    anchor = min(
        (lowered.find(normalise(k)) for k in keywords if normalise(k) in lowered), default=0
    )
    start = max(0, min(anchor - 60, len(sentence) - MAX_EVIDENCE_CHARS))
    piece = sentence[start : start + MAX_EVIDENCE_CHARS]
    if start > 0 and " " in piece:
        piece = piece.split(" ", 1)[1]
    if start + MAX_EVIDENCE_CHARS < len(sentence) and " " in piece:
        piece = piece.rsplit(" ", 1)[0]
    return piece.strip()


def find_matches(documents: list[Document], source: str, keywords: list[str]) -> list[dict]:
    """Sentences (from documents whose name contains ``source``) that hold every keyword."""
    needles = [normalise(k) for k in keywords]
    found = []
    for doc in documents:
        if source and normalise(source) not in normalise(doc.source):
            continue
        for sentence in split_sentences(doc.text):
            flat = normalise(sentence)
            if all(n in flat for n in needles):
                found.append({"source": doc.source, "sentence": " ".join(sentence.split())})
    return found


def build_entries(documents: list[Document], specs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return ``(entries, problems)``; a problem is a spec with no usable match."""
    entries, problems = [], []
    for spec in specs:
        matches = find_matches(documents, spec.get("source", ""), spec["all"])
        usable = [m for m in matches if len(m["sentence"]) >= MIN_EVIDENCE_CHARS]
        if not usable:
            problems.append({"question": spec["question"], "reason": "no sentence matches"})
            continue
        best = min(usable, key=lambda m: len(m["sentence"]))
        entries.append(
            {
                "question": spec["question"],
                "evidence": clip_evidence(best["sentence"], spec["all"]),
                "source": best["source"],
                "notes": f"spec-matched ({len(usable)} candidate sentences; review)",
            }
        )
    return entries, problems


def append_new(path: Path, entries: list[dict]) -> int:
    """Append entries whose question is not already in the gold file; return how many."""
    existing = {normalise(g["question"]) for g in load_gold(path)} if path.exists() else set()
    fresh = [e for e in entries if normalise(e["question"]) not in existing]
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for entry in fresh:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(fresh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--gold", default=str(GOLD_PATH))
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--write", action="store_true", help="append to the gold file")
    args = parser.parse_args()

    specs = [
        json.loads(line)
        for line in Path(args.spec).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries, problems = build_entries(load_corpus(Path(args.corpus_dir)), specs)
    for number, entry in enumerate(entries, 1):
        print(f"[{number}] {entry['question']}")
        print(f"    {re.sub(r'ley_|directiva_', '', entry['source'])[:12]}: {entry['evidence']}\n")
    for problem in problems:
        print(f"NO MATCH: {problem['question']}  (change the keywords)")
    print(f"{len(entries)} matched, {len(problems)} without match")
    if args.write:
        print(f"appended {append_new(Path(args.gold), entries)} new entries to {args.gold}")


if __name__ == "__main__":
    main()
