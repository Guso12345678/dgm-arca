"""Dense and hybrid retrievers, with a deterministic hashing embedder."""

import numpy as np
import pytest
from rag_helpers import HashEmbedder, make_documents

from rag.chunking import recursive_chunks
from rag.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    corpus_fingerprint,
    min_max,
)


@pytest.fixture
def chunks():
    return recursive_chunks(make_documents(), chunk_size=200, chunk_overlap=0)


def test_min_max():
    assert min_max([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert min_max([5.0, 5.0]) == [1.0, 1.0]  # signal, no spread
    assert min_max([0.0, 0.0]) == [0.0, 0.0]  # no signal must stay no signal
    assert min_max([2.0]) == [1.0]


def test_dense_returns_cosine_ranked_hits(chunks):
    dense = DenseRetriever(chunks, HashEmbedder(), persist_dir=None)
    hits = dense.search("derecho de desistimiento catorce días", top_k=3)
    assert len(hits) == 3
    assert "desistimiento" in hits[0].chunk.text
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(-1.0 <= h.score <= 1.0 + 1e-9 for h in hits)
    expected = dense.matrix @ dense._embed_query("derecho de desistimiento catorce días")
    assert hits[0].score == pytest.approx(expected.max())


def test_dense_persists_and_reuses_the_index(chunks, tmp_path):
    first = HashEmbedder()
    DenseRetriever(chunks, first, persist_dir=tmp_path)
    assert first.passage_calls == 1
    assert len(list(tmp_path.glob("dense_*.npz"))) == 1

    second = HashEmbedder()
    reloaded = DenseRetriever(chunks, second, persist_dir=tmp_path)
    assert second.passage_calls == 0  # loaded from disk, not re-embedded
    assert reloaded.search("usura interés", top_k=1)[0].chunk.source == "usura.txt"


def test_a_different_chunking_gets_its_own_index(chunks, tmp_path):
    other = recursive_chunks(make_documents(), chunk_size=400, chunk_overlap=0)
    assert corpus_fingerprint(chunks, "m") != corpus_fingerprint(other, "m")
    assert corpus_fingerprint(chunks, "m") != corpus_fingerprint(chunks, "other-model")
    DenseRetriever(chunks, HashEmbedder(), persist_dir=tmp_path)
    fresh = HashEmbedder()
    DenseRetriever(other, fresh, persist_dir=tmp_path)
    assert fresh.passage_calls == 1
    assert len(list(tmp_path.glob("dense_*.npz"))) == 2


def test_instruction_reaches_the_embedder_and_queries_are_cached(chunks):
    embedder = HashEmbedder()
    dense = DenseRetriever(chunks, embedder, persist_dir=None, instruction="Retrieve law.")
    dense.search("tae", top_k=1)
    assert embedder.last_instruction == "Retrieve law."
    embedder.last_instruction = None
    dense.search("tae", top_k=1)
    assert embedder.last_instruction is None  # second call served from the cache


def test_default_instruction_is_not_forced(chunks):
    embedder = HashEmbedder()
    DenseRetriever(chunks, embedder, persist_dir=None).search("tae", top_k=1)
    assert embedder.last_instruction == "default"  # the embedder's own default was used


def test_hybrid_extremes_reproduce_the_single_retrievers(chunks):
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, HashEmbedder(), persist_dir=None)
    query = "evaluará la solvencia del consumidor"
    hybrid = HybridRetriever(bm25, dense, weight_bm25=1.0, pool=len(chunks))
    ids_bm25 = [h.chunk.id for h in bm25.search(query, top_k=len(chunks))][:3]
    assert [h.chunk.id for h in hybrid.search(query, top_k=3)][0] == ids_bm25[0]

    hybrid.weight_bm25 = 0.0
    ids_dense = [h.chunk.id for h in dense.search(query, top_k=3)]
    assert [h.chunk.id for h in hybrid.search(query, top_k=3)][0] == ids_dense[0]


def test_hybrid_scores_are_in_unit_interval_and_ids_unique(chunks):
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, HashEmbedder(), persist_dir=None)
    hits = HybridRetriever(bm25, dense, weight_bm25=0.4, pool=10).search("reembolso anticipado", 8)
    assert len(hits) == len({h.chunk.id for h in hits})
    assert all(0.0 <= h.score <= 1.0 + 1e-9 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert "reembolso" in hits[0].chunk.text


def test_hybrid_rescues_what_bm25_misses():
    # The query shares no word with the passage that answers it in BM25 terms, but the
    # (fake) dense side knows it: the hybrid must still return it.
    from rag.chunking import Chunk

    chunks = [
        Chunk(f"c{i}", "d", "s.txt", text, i)
        for i, text in enumerate(["alfa beta gamma", "delta epsilon zeta", "eta theta iota"])
    ]

    class Oracle:
        model_id = "oracle"

        def embed_passages(self, texts, batch_size=32):
            return np.eye(3)

        def embed_queries(self, texts, instruction=None):
            return np.array([[0.0, 1.0, 0.0]])  # points at chunk c1

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, Oracle(), persist_dir=None)
    hybrid = HybridRetriever(bm25, dense, weight_bm25=0.3, pool=3)
    assert bm25.search("alfa", top_k=1)[0].chunk.id == "c0"
    hits = hybrid.search("alfa", top_k=3)
    ids = [h.chunk.id for h in hits]
    assert {"c0", "c1"} <= set(ids)
