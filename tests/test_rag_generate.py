"""Prompt, citation checks, abstention and the pipeline with a fake generator."""

import pytest
from rag_helpers import HashEmbedder, make_documents

from rag.chunking import recursive_chunks
from rag.generate import (
    NO_INFO_ANSWER,
    RagPipeline,
    build_prompt,
    check_citations,
    strip_invalid_citations,
)
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

VALID = {"abcdef0123-0001", "abcdef0123-0002"}


def test_check_citations_counts_only_chunk_shaped_ids():
    answer = "Sí [abcdef0123-0001], también [abcdef0123-0099] y [1] y [sic]."
    assert check_citations(answer, VALID) == {"cited": 2, "invalid": 1}


def test_strip_invalid_citations_keeps_valid_and_non_chunk_brackets():
    answer = "La TAE [abcdef0123-0001] incluye gastos [abcdef0123-0099] . Ver [1]."
    cleaned = strip_invalid_citations(answer, VALID)
    assert "[abcdef0123-0099]" not in cleaned
    assert "[abcdef0123-0001]" in cleaned and "[1]" in cleaned
    assert "  " not in cleaned and " ." not in cleaned


@pytest.fixture
def pipeline_parts():
    chunks = recursive_chunks(make_documents(), 200, 0)
    embedder = HashEmbedder()
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embedder, persist_dir=None)
    hybrid = HybridRetriever(bm25, dense)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}


def test_prompt_contains_ids_sources_and_question(pipeline_parts):
    hits = pipeline_parts["dense"].search("desistimiento", top_k=2)
    prompt = build_prompt("¿Qué plazo?", hits)
    assert f"[{hits[0].chunk.id}]" in prompt and hits[0].chunk.source in prompt
    assert prompt.rstrip().endswith("¿Qué plazo?")


def test_answer_returns_valid_citations_only_and_every_retriever_works(pipeline_parts):
    calls = []

    def generate(messages):
        calls.append(messages)
        chunk_id = messages[-1]["content"].split("[")[2].split("]")[0]
        return f"El plazo es de catorce días [{chunk_id}] [ffffffffff-9999]."

    pipeline = RagPipeline.from_components(pipeline_parts, generate, model_name="fake-1")
    for name in ("bm25", "dense", "hybrid"):
        response = pipeline.answer("derecho de desistimiento catorce días", top_k=3, retriever=name)
        ids = {c.id for c in response.chunks}
        assert response.retriever == name and response.model == "fake-1"
        assert len(response.chunks) == 3
        assert response.citations and set(response.citations) <= ids
        assert "ffffffffff-9999" not in response.answer  # invented ids never reach the user
    assert calls[0][0]["role"] == "system" and NO_INFO_ANSWER in calls[0][0]["content"]


def test_unknown_retriever_is_a_clear_error(pipeline_parts):
    pipeline = RagPipeline.from_components(pipeline_parts, lambda m: "x")
    with pytest.raises(ValueError, match="retriever must be one of"):
        pipeline.answer("pregunta", retriever="magic")


def test_similarity_gate_abstains_without_calling_the_model(pipeline_parts):
    def never(messages):
        raise AssertionError("the generator must not be called")

    pipeline = RagPipeline.from_components(pipeline_parts, never, min_similarity=0.5)
    response = pipeline.answer("fútbol campeonato goles", retriever="dense")
    assert response.answer == NO_INFO_ANSWER and response.citations == []
    assert response.chunks  # what was retrieved is still reported


def test_gate_lets_answerable_questions_through(pipeline_parts):
    pipeline = RagPipeline.from_components(
        pipeline_parts, lambda m: "Respuesta.", min_similarity=0.3
    )
    assert pipeline.is_answerable("derecho de desistimiento en catorce días naturales")
    assert pipeline.answer("derecho de desistimiento catorce días").answer == "Respuesta."


def test_gate_is_off_by_default(pipeline_parts):
    pipeline = RagPipeline.from_components(pipeline_parts, lambda m: "ok")
    assert pipeline.min_similarity == 0.0
    assert pipeline.is_answerable("fútbol campeonato goles")


def test_the_model_can_abstain_by_itself(pipeline_parts):
    pipeline = RagPipeline.from_components(pipeline_parts, lambda m: NO_INFO_ANSWER)
    response = pipeline.answer("fútbol campeonato goles")
    assert response.answer == NO_INFO_ANSWER and response.citations == []


def test_load_without_corpus_explains_what_to_do(tmp_path, monkeypatch):
    import rag.ingest as ingest

    monkeypatch.setattr(ingest, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(ingest, "load_corpus", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="fetch_corpus"):
        RagPipeline().load()
