"""Plot the distance between consecutive sentences of a document, with candidate thresholds.

This is the figure that justifies the semantic chunker's threshold (as in the slides): the
chunker cuts wherever the curve goes above the line. It also prints the percentiles of the
distances, because a threshold is easier to reason about as "cut at the 80 % most abrupt
changes" than as a bare number.

    uv run python -m rag.plot_distances --doc ley_16_2011 --thresholds 0.25 0.35 \\
        --out reports/phase3_distances.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def percentile_table(distances: np.ndarray, thresholds: list[float]) -> list[dict]:
    """For each threshold, the share of sentence boundaries that would become a cut."""
    return [
        {"threshold": t, "share_of_cuts": float(np.mean(distances > t)) if len(distances) else 0.0}
        for t in thresholds
    ]


def percentile_thresholds(distances: np.ndarray, percentiles: list[float]) -> dict[float, float]:
    """The distance value at each percentile: cutting there cuts ``100 - p`` % of boundaries."""
    if not len(distances):
        return {p: 0.0 for p in percentiles}
    return {p: float(np.percentile(distances, p)) for p in percentiles}


def plot_distances(distances: np.ndarray, thresholds: list[float], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.arange(len(distances)), distances, lw=0.9, color="#3366aa")
    for threshold in thresholds:
        ax.axhline(threshold, ls="--", lw=1, color="#cc4444")
        ax.text(len(distances) * 1.005, threshold, f"{threshold:g}", va="center", color="#cc4444")
    ax.set_xlabel("sentence boundary (i → i+1)")
    ax.set_ylabel("cosine distance")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default=None, help="substring of the document's file name")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.6, 0.7])
    parser.add_argument("--percentiles", type=float, nargs="*", default=[70, 80])
    parser.add_argument("--max-sentences", type=int, default=400)
    parser.add_argument("--out", default="reports/phase3_distances.png")
    args = parser.parse_args()

    from rag.chunking import sentence_distances, split_sentences
    from rag.embed import Embedder
    from rag.ingest import load_corpus

    documents = load_corpus()
    if args.doc:
        documents = [d for d in documents if args.doc in d.source]
    if not documents:
        raise SystemExit("no matching document in rag/corpus/")
    doc = max(documents, key=lambda d: len(d.text)) if not args.doc else documents[0]
    sentences = split_sentences(doc.text)[: args.max_sentences]
    distances = sentence_distances(Embedder().embed_passages(sentences))

    print(f"{doc.source}: {len(sentences)} sentences")
    print(f"distance mean {distances.mean():.3f}, median {np.median(distances):.3f}")
    for row in percentile_table(distances, args.thresholds):
        print(
            f"  threshold {row['threshold']:g}: cuts {row['share_of_cuts']:.0%} of the boundaries"
        )
    at_percentile = percentile_thresholds(distances, args.percentiles)
    for p, value in at_percentile.items():
        print(f"  percentile {p:g} -> distance {value:.3f} (cuts {100 - p:g}% of the boundaries)")
    plot_distances(
        distances,
        args.thresholds + list(at_percentile.values()),
        Path(args.out),
        f"Consecutive-sentence distance: {doc.source}",
    )
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
