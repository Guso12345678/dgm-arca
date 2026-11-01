"""Retrieval: similarity search, keyword search (BM25) and the hybrid of both.

``BM25Retriever`` is complete: it is the keyword baseline you compare against.
``DenseRetriever`` and ``HybridRetriever`` are yours. All three return the same ``Hit``
objects so that ``rag/generate.py`` and ``rag/evaluate.py`` do not care which one is used.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Keyword ranking: term frequency, inverse document frequency, length normalisation."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def corpus_fingerprint(chunks: list[Chunk], model_id: str) -> str:
    """Hash of the embedding model and of every chunk (id and text).

    It names the index file, so a different corpus, chunking or model never reuses a stale
    index: the file simply does not exist and is rebuilt.
    """
    digest = hashlib.sha256(model_id.encode())
    for chunk in chunks:
        digest.update(chunk.id.encode())
        digest.update(b"\0")
        digest.update(chunk.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


class DenseRetriever:
    """Embed the chunks once, store them, embed the query and rank by cosine similarity.

    The store is a plain NumPy matrix (rows are L2-normalised, so cosine = dot product).
    With a few thousand chunks an exact search is instantaneous and a vector database would
    add a dependency without adding anything. It is saved as
    ``<persist_dir>/dense_<fingerprint>.npz``: the next start loads it instead of re-embedding
    the corpus, and any change of corpus, chunking or model produces a new file name.

    ``instruction`` is the task description that goes in front of every query (see
    ``rag/embed.py``); ``None`` keeps the embedder's default. It only affects queries, so
    changing it never invalidates the index.
    """

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        persist_dir: str | Path | None = "rag/index",
        instruction: str | None = None,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.instruction = instruction
        self._query_cache: dict[tuple[str | None, str], np.ndarray] = {}
        self.matrix = self._load_or_build(persist_dir)

    def _load_or_build(self, persist_dir: str | Path | None) -> np.ndarray:
        model_id = getattr(self.embedder, "model_id", type(self.embedder).__name__)
        path = None
        if persist_dir is not None:
            fingerprint = corpus_fingerprint(self.chunks, model_id)
            path = Path(persist_dir) / f"dense_{fingerprint}.npz"
            if path.exists():
                return np.load(path)["embeddings"]
        matrix = np.asarray(self.embedder.embed_passages([c.text for c in self.chunks]))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, embeddings=matrix.astype(np.float32))
        return matrix

    def _embed_query(self, query: str) -> np.ndarray:
        key = (self.instruction, query)
        if key not in self._query_cache:
            if self.instruction is None:
                vector = self.embedder.embed_queries([query])
            else:
                vector = self.embedder.embed_queries([query], instruction=self.instruction)
            self._query_cache[key] = np.asarray(vector)[0]
        return self._query_cache[key]

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query with every chunk."""
        return self.matrix @ self._embed_query(query)

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def min_max(values: list[float]) -> list[float]:
    """Rescale to [0, 1]. A flat list maps to 1.0 if it carries signal (> 0) and to 0.0 if not.

    The second case is BM25 with no word in common with the query: all scores are 0 and
    they must not be promoted to a perfect match.
    """
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


class HybridRetriever:
    """s_hybrid = λ · s_bm25 + (1 - λ) · s_dense, on scores rescaled to [0, 1].

    Take the top ``pool`` hits from each retriever, min-max normalise each score list so
    they are comparable, merge by chunk id, and return the top ``top_k`` by the weighted
    sum. A chunk that only one retriever put in its pool gets 0 from the other one.
    ``weight_bm25`` is λ: λ=1 is BM25 only, λ=0 dense only. It is a plain attribute so the
    λ sweep in ``rag/evaluate.py`` can change it without rebuilding anything.
    """

    name = "hybrid"

    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, weight_bm25: float = 0.5, pool: int = 20
    ):
        self.bm25, self.dense, self.weight_bm25, self.pool = bm25, dense, weight_bm25, pool

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        merged: dict[str, dict] = {}
        for source, retriever in (("bm25", self.bm25), ("dense", self.dense)):
            hits = retriever.search(query, top_k=self.pool)
            if not hits:
                continue
            for hit, value in zip(hits, min_max([h.score for h in hits]), strict=True):
                entry = merged.setdefault(
                    hit.chunk.id, {"chunk": hit.chunk, "bm25": 0.0, "dense": 0.0}
                )
                entry[source] = value
        lam = self.weight_bm25
        ranked = sorted(
            merged.values(),
            key=lambda e: (-(lam * e["bm25"] + (1 - lam) * e["dense"]), -e["dense"], e["chunk"].id),
        )
        return [Hit(e["chunk"], lam * e["bm25"] + (1 - lam) * e["dense"]) for e in ranked[:top_k]]
