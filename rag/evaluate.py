"""Phase 3 evaluation: Recall@k and MRR of each retriever on your gold set.

The gold set is a JSONL file (``rag/gold/example_gold.jsonl`` shows the format) with a
question and the id(s) of the chunk(s) that answer it. Fifty questions is the minimum;
write them yourselves while reading the corpus, and note the ones where two chunks are
equally valid: they will tell you a lot about your chunking.

Chunk ids depend on the chunking strategy (``<doc>-0007`` is a different text in the
recursive and in the semantic chunker), so a gold set made of ids could only judge one of
them. Instead each gold item can carry the **evidence**: a short verbatim passage of the
corpus that answers the question (``"evidence"``, optionally with ``"source"`` = file name).
``resolve_gold`` turns it into the ids of the chunks that contain it, for each strategy.
Items with explicit ``"relevant_chunk_ids"`` still work. Check your file with::

    uv run python -m rag.evaluate --check-gold --gold rag/gold/gold.jsonl

(no GPU needed), and run the full comparison with::

    uv run python -m rag.evaluate --gold rag/gold/gold.jsonl --k 5

Optional steps: ``--calibrate rag/gold/unanswerable.jsonl`` suggests the similarity gate for
abstention; ``--generation`` runs the generator over the gold questions and measures
citations and abstention.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

QUERY_INSTRUCTIONS = {
    "default": None,  # the embedder's own default
    "minimal": "Retrieve relevant passages.",
    "domain": (
        "Given a question from a credit-risk analyst about consumer-credit regulation, "
        "retrieve the legal or supervisory passages that answer it."
    ),
}
LAMBDAS = [round(0.1 * i, 1) for i in range(11)]


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / position of the first relevant chunk (0 when none is retrieved)."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def evaluate_retriever(retriever, gold: list[dict], k: int) -> dict[str, float]:
    recalls, rrs = [], []
    for item in gold:
        ranked = [hit.chunk.id for hit in retriever.search(item["question"], top_k=max(k, 10))]
        relevant = set(item["relevant_chunk_ids"])
        recalls.append(recall_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
    n = max(len(gold), 1)
    return {f"recall@{k}": sum(recalls) / n, "mrr": sum(rrs) / n, "n_questions": len(gold)}


def load_gold(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------------------
# Gold set <-> chunks
# ---------------------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lower-case, NFKC, no soft hyphens, single spaces: so evidence typed by hand matches."""
    text = unicodedata.normalize("NFKC", text).replace("­", "").lower()
    return re.sub(r"\s+", " ", text).strip()


def resolve_gold(gold: list[dict], chunks: list) -> tuple[list[dict], list[dict]]:
    """Fill ``relevant_chunk_ids`` for one chunking. Returns ``(resolved, unmatched)``.

    An item with ``evidence`` is relevant to every chunk whose text contains it (so a passage
    that falls in the overlap of two chunks makes both valid). An item with only explicit ids
    keeps those that exist. Unmatched items (evidence split across a chunk boundary, or ids
    that this chunking does not have) are returned apart, never silently scored as misses.
    """
    texts = [(chunk, normalise(chunk.text)) for chunk in chunks]
    existing = {chunk.id for chunk in chunks}
    resolved, unmatched = [], []
    for item in gold:
        if item.get("evidence"):
            needle, source = normalise(item["evidence"]), item.get("source")
            ids = [
                chunk.id
                for chunk, text in texts
                if needle in text and (source is None or chunk.source.endswith(source))
            ]
        else:
            ids = [i for i in item.get("relevant_chunk_ids", []) if i in existing]
        (resolved if ids else unmatched).append({**item, "relevant_chunk_ids": ids})
    return resolved, unmatched


# ---------------------------------------------------------------------------------------
# λ selection and abstention threshold
# ---------------------------------------------------------------------------------------


def sweep_lambda(hybrid, gold: list[dict], k: int, lambdas: list[float] = LAMBDAS) -> list[dict]:
    """Recall@k and MRR of the hybrid retriever for every λ (weight of BM25)."""
    rows = []
    original = hybrid.weight_bm25
    try:
        for lam in lambdas:
            hybrid.weight_bm25 = lam
            rows.append({"lambda": lam, **evaluate_retriever(hybrid, gold, k)})
    finally:
        hybrid.weight_bm25 = original
    return rows


def best_lambda(rows: list[dict]) -> float:
    """λ with the highest MRR; ties go to the one closest to 0.5 (the least extreme blend)."""
    return max(rows, key=lambda r: (round(r["mrr"], 9), -abs(r["lambda"] - 0.5), -r["lambda"]))[
        "lambda"
    ]


def cross_validated_lambda(hybrid, gold: list[dict], k: int) -> dict[str, float]:
    """Choose λ on half of the questions and score it on the other half (and vice versa).

    Picking λ on the same questions that are then reported is optimistic; with fifty
    questions the honest number is the one measured on questions λ has not seen.
    """
    folds = [gold[0::2], gold[1::2]]
    mrrs, recalls, chosen = [], [], []
    for train, test in ((0, 1), (1, 0)):
        lam = best_lambda(sweep_lambda(hybrid, folds[train], k))
        original, hybrid.weight_bm25 = hybrid.weight_bm25, lam
        result = evaluate_retriever(hybrid, folds[test], k)
        hybrid.weight_bm25 = original
        chosen.append(lam)
        mrrs.append(result["mrr"])
        recalls.append(result[f"recall@{k}"])
    return {
        "lambdas_chosen": chosen,
        "mrr_heldout": float(np.mean(mrrs)),
        f"recall@{k}_heldout": float(np.mean(recalls)),
    }


def suggest_threshold(answerable: list[float], unanswerable: list[float]) -> dict[str, float]:
    """Similarity gate for abstention from the best-chunk cosine of each kind of question.

    ``balanced`` maximises (recall of answerable + recall of unanswerable) / 2;
    ``keep_all`` is the highest threshold that still answers every answerable question.
    """
    candidates = sorted(set(answerable) | set(unanswerable))
    best, best_score = 0.0, -1.0
    for t in candidates:
        score = (np.mean([s >= t for s in answerable]) + np.mean([s < t for s in unanswerable])) / 2
        if score > best_score:
            best, best_score = t, float(score)
    return {"balanced": float(best), "balanced_accuracy": best_score, "keep_all": min(answerable)}


# ---------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------


def cached(embed_fn):
    """Memoise sentence embeddings: every semantic strategy re-embeds the same sentences."""
    store: dict[str, object] = {}

    def embed(texts):
        missing = [t for t in dict.fromkeys(texts) if t not in store]
        if missing:
            for text, vector in zip(missing, embed_fn(missing), strict=True):
                store[text] = vector
        return np.stack([store[t] for t in texts])

    return embed


def build_strategies(
    documents, embedder, chunk_size, overlap, thresholds, max_chars, percentiles=()
):
    """Recursive baseline plus one semantic strategy per fixed threshold and per percentile."""
    from rag.chunking import recursive_chunks, semantic_chunks

    strategies = {f"recursive-{chunk_size}": recursive_chunks(documents, chunk_size, overlap)}
    embed = cached(embedder.embed_passages)
    for threshold in thresholds:
        strategies[f"semantic-{threshold:g}"] = semantic_chunks(
            documents, embed, threshold, max_chars
        )
    for percentile in percentiles:
        strategies[f"semantic-p{percentile:g}"] = semantic_chunks(
            documents, embed, max_chunk_chars=max_chars, threshold_percentile=percentile
        )
    return strategies


def run_strategy(chunks, gold, embedder, k, persist_dir) -> dict:
    """BM25, dense (one row per query instruction) and hybrid (λ sweep) on one chunking."""
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    resolved, unmatched = resolve_gold(gold, chunks)
    result: dict = {"n_matched": len(resolved), "n_unmatched": len(unmatched)}
    if not resolved:
        return result
    bm25 = BM25Retriever(chunks)
    result["bm25"] = evaluate_retriever(bm25, resolved, k)

    result["dense_by_instruction"] = {}
    dense = None
    for name, instruction in QUERY_INSTRUCTIONS.items():
        dense = DenseRetriever(chunks, embedder, persist_dir=persist_dir, instruction=instruction)
        result["dense_by_instruction"][name] = evaluate_retriever(dense, resolved, k)
    dense = DenseRetriever(chunks, embedder, persist_dir=persist_dir)
    result["dense"] = result["dense_by_instruction"]["default"]

    hybrid = HybridRetriever(bm25, dense)
    result["lambda_sweep"] = sweep_lambda(hybrid, resolved, k)
    result["best_lambda"] = best_lambda(result["lambda_sweep"])
    result["lambda_cv"] = cross_validated_lambda(hybrid, resolved, k)
    return result


def print_summary(results: dict, k: int) -> None:
    print("\n== Chunking ==")
    header = ("strategy", "chunks", "mean", "min", "max", "evaluated", "lost")
    print(f"{header[0]:<20}" + "".join(f"{h:>10}" for h in header[1:]))
    for name, entry in results["strategies"].items():
        d = entry["chunking"]
        print(
            f"{name:<20}{d['n_chunks']:>10}{d['mean_chars']:>10.0f}{d['min_chars']:>10}"
            f"{d['max_chars']:>10}{entry['n_matched']:>10}{entry['n_unmatched']:>10}"
        )

    def cell(row: dict) -> str:
        return f"{row[f'recall@{k}']:.3f} / {row['mrr']:.3f}"

    print(f"\n== Retrievers (Recall@{k} / MRR) ==")
    print(f"{'strategy':<20}{'bm25':>16}{'dense':>16}{'hybrid (best λ)':>26}")
    for name, entry in results["strategies"].items():
        if "bm25" not in entry:
            continue
        lam = entry["best_lambda"]
        best = next(r for r in entry["lambda_sweep"] if r["lambda"] == lam)
        hybrid_cell = f"{cell(best)} (λ={lam:g})"
        print(f"{name:<20}{cell(entry['bm25']):>16}{cell(entry['dense']):>16}{hybrid_cell:>26}")


def calibrate(dense, answerable: list[dict], unanswerable: list[dict]) -> dict:
    ok = [float(dense.scores(i["question"]).max()) for i in answerable]
    no = [float(dense.scores(i["question"]).max()) for i in unanswerable]
    return {
        "answerable": {"mean": float(np.mean(ok)), "min": min(ok), "max": max(ok)},
        "unanswerable": {"mean": float(np.mean(no)), "min": min(no), "max": max(no)},
        **suggest_threshold(ok, no),
    }


def evaluate_generation(pipeline, answerable: list[dict], unanswerable: list[dict], k: int) -> dict:
    """Run the generator on gold questions: citations, grounding in a relevant chunk, abstention."""
    from rag.generate import NO_INFO_ANSWER, check_citations

    def run(item: dict) -> dict:
        hits = pipeline.retrieve(item["question"], k, "hybrid")
        ids = {h.chunk.id for h in hits}
        if not pipeline.is_answerable(item["question"]):
            raw = NO_INFO_ANSWER
        else:
            raw = pipeline.generate_raw(item["question"], hits)
        stats = check_citations(raw, ids)
        relevant = set(item.get("relevant_chunk_ids", []))
        cited = {m for m in re.findall(r"\[([A-Za-z0-9_-]+)\]", raw)} & ids
        return {
            "question": item["question"],
            "answer": raw,
            "abstained": NO_INFO_ANSWER[:25].lower() in raw.lower(),
            "cited": stats["cited"],
            "invalid_citations": stats["invalid"],
            "cites_relevant": bool(cited & relevant),
            "retrieved_relevant": bool(ids & relevant),
        }

    rows_a = [run(i) for i in answerable]
    rows_u = [run(i) for i in unanswerable]
    n_a, n_u = max(len(rows_a), 1), max(len(rows_u), 1)
    return {
        "answerable": {
            "n": len(rows_a),
            "with_citation": sum(r["cited"] > 0 for r in rows_a) / n_a,
            "invalid_citation_rate": sum(r["invalid_citations"] > 0 for r in rows_a) / n_a,
            "cites_relevant_chunk": sum(r["cites_relevant"] for r in rows_a) / n_a,
            "wrongly_abstained": sum(r["abstained"] for r in rows_a) / n_a,
        },
        "unanswerable": {"n": len(rows_u), "abstained": sum(r["abstained"] for r in rows_u) / n_u},
        "rows": rows_a + rows_u,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="rag/gold/example_gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="reports/phase3_eval.json")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument(
        "--semantic-thresholds",
        type=float,
        nargs="*",
        default=[0.7],
        help="fixed distance thresholds (Qwen3-Embedding distances are high: see plot_distances)",
    )
    parser.add_argument(
        "--semantic-percentiles",
        type=float,
        nargs="*",
        default=[70, 80],
        help="cut at this percentile of each document's own distances",
    )
    parser.add_argument("--max-chunk-chars", type=int, default=1500)
    parser.add_argument("--index-dir", default="rag/index")
    parser.add_argument("--check-gold", action="store_true", help="validate the gold set only")
    parser.add_argument("--calibrate", metavar="UNANSWERABLE_JSONL")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--unanswerable", default="rag/gold/unanswerable.jsonl")
    parser.add_argument("--chunking", default=None, help="strategy for --calibrate/--generation")
    args = parser.parse_args()

    from rag.chunking import describe, recursive_chunks
    from rag.ingest import load_corpus

    gold = load_gold(args.gold)
    documents = load_corpus()
    print(f"{len(gold)} gold questions, {len(documents)} documents")

    if args.check_gold:
        chunks = recursive_chunks(documents, args.chunk_size, args.overlap)
        resolved, unmatched = resolve_gold(gold, chunks)
        print(f"recursive-{args.chunk_size}: {len(resolved)} matched, {len(unmatched)} unmatched")
        for item in unmatched:
            print("  UNMATCHED:", item["question"])
        multi = sum(len(i["relevant_chunk_ids"]) > 1 for i in resolved)
        print(f"{multi} questions have more than one valid chunk (overlap / repeated text)")
        return

    from rag.embed import Embedder

    embedder = Embedder()
    strategies = build_strategies(
        documents,
        embedder,
        args.chunk_size,
        args.overlap,
        args.semantic_thresholds,
        args.max_chunk_chars,
        args.semantic_percentiles,
    )
    # Fair comparison: every strategy is scored on the questions whose evidence resolves under
    # ALL of them. How many each strategy loses on its own is reported as a result in itself
    # (evidence cut by a chunk boundary is a cost of that chunking).
    unmatched_own, common = {}, None
    for name, chunks in strategies.items():
        resolved, unmatched = resolve_gold(gold, chunks)
        unmatched_own[name] = len(unmatched)
        questions = {item["question"] for item in resolved}
        common = questions if common is None else common & questions
    common_gold = [item for item in gold if item["question"] in (common or set())]
    print(f"{len(common_gold)} of {len(gold)} questions resolve under every strategy")

    results: dict = {
        "k": args.k,
        "n_gold": len(gold),
        "n_common": len(common_gold),
        "strategies": {},
    }
    for name, chunks in strategies.items():
        print(f"evaluating {name} ({len(chunks)} chunks)...", flush=True)
        entry = run_strategy(chunks, common_gold, embedder, args.k, args.index_dir)
        entry["n_unmatched"] = unmatched_own[name]
        entry["chunking"] = describe(chunks)
        results["strategies"][name] = entry
    print_summary(results, args.k)

    chosen = args.chunking or next(iter(strategies))
    if args.calibrate or args.generation:
        from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

        chunks = strategies[chosen]
        resolved, _ = resolve_gold(gold, chunks)  # (the chosen strategy's own questions)
        unanswerable_path = args.calibrate or args.unanswerable
        unanswerable = load_gold(unanswerable_path)
        dense = DenseRetriever(chunks, embedder, persist_dir=args.index_dir)
        if args.calibrate:
            results["calibration"] = calibrate(dense, resolved, unanswerable)
            print("\n== Abstention gate ==\n", json.dumps(results["calibration"], indent=2))
        if args.generation:
            import os

            from rag.generate import DEFAULT_GENERATOR, RagPipeline, load_chat_generator

            bm25 = BM25Retriever(chunks)
            lam = results["strategies"][chosen].get("best_lambda", 0.5)
            hybrid = HybridRetriever(bm25, dense, weight_bm25=lam)
            model = os.environ.get("ARCA_RAG_MODEL", DEFAULT_GENERATOR)
            pipeline = RagPipeline.from_components(
                {"bm25": bm25, "dense": dense, "hybrid": hybrid},
                load_chat_generator(model, 400),
                model,
                float(os.environ.get("ARCA_RAG_MIN_SIMILARITY", 0.0)),
            )
            results["generation"] = evaluate_generation(pipeline, resolved, unanswerable, args.k)
            summary = {k: v for k, v in results["generation"].items() if k != "rows"}
            print("\n== Generation ==\n", json.dumps(summary, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
