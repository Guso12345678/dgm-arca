"""Gold resolution, λ selection, abstention threshold and an end-to-end strategy run."""

import pytest
from rag_helpers import HashEmbedder, make_documents

from rag.chunking import Chunk, recursive_chunks, semantic_chunks
from rag.evaluate import (
    best_lambda,
    calibrate,
    cross_validated_lambda,
    evaluate_generation,
    evaluate_retriever,
    normalise,
    resolve_gold,
    run_strategy,
    suggest_threshold,
    sweep_lambda,
)
from rag.generate import NO_INFO_ANSWER, RagPipeline
from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever


def chunk(i, text, source="a.txt"):
    return Chunk(f"c{i}", "d", source, text, i)


def test_normalise_ignores_case_spacing_and_soft_hyphens():
    assert normalise("  La  TAE\n incluye­gastos ") == "la tae incluyegastos"


def test_resolve_gold_by_evidence_covers_every_chunk_that_contains_it():
    chunks = [chunk(0, "uno dos tres cuatro"), chunk(1, "tres cuatro cinco"), chunk(2, "seis")]
    gold = [{"question": "q", "evidence": "Tres  CUATRO"}]
    resolved, unmatched = resolve_gold(gold, chunks)
    assert resolved[0]["relevant_chunk_ids"] == ["c0", "c1"]  # the overlap makes both valid
    assert unmatched == []


def test_resolve_gold_reports_unmatched_instead_of_scoring_a_miss():
    chunks = [chunk(0, "uno dos"), chunk(1, "tres cuatro")]
    gold = [
        {"question": "split", "evidence": "dos tres"},  # crosses a chunk boundary
        {"question": "ghost", "relevant_chunk_ids": ["zzz-0001"]},
        {"question": "ok", "relevant_chunk_ids": ["c1", "zzz-0001"]},
    ]
    resolved, unmatched = resolve_gold(gold, chunks)
    assert [i["question"] for i in unmatched] == ["split", "ghost"]
    assert resolved[0]["relevant_chunk_ids"] == ["c1"]


def test_resolve_gold_source_disambiguates_repeated_text():
    chunks = [chunk(0, "texto repetido", "a.txt"), chunk(1, "texto repetido", "b.txt")]
    resolved, _ = resolve_gold(
        [{"question": "q", "evidence": "texto repetido", "source": "b.txt"}], chunks
    )
    assert resolved[0]["relevant_chunk_ids"] == ["c1"]


def test_the_same_gold_resolves_under_both_chunkings():
    docs = make_documents()
    embedder = HashEmbedder()
    gold = [{"question": "¿Cuántos días?", "evidence": "catorce días naturales"}]
    for chunks in (
        recursive_chunks(docs, 200, 0),
        semantic_chunks(docs, embedder.embed_passages, 0.4, 600),
    ):
        resolved, unmatched = resolve_gold(gold, chunks)
        assert resolved and not unmatched
        assert all(
            "desistimiento" in c.text for c in chunks if c.id in resolved[0]["relevant_chunk_ids"]
        )


def rows(*pairs):
    return [{"lambda": lam, "mrr": mrr, "recall@5": 1.0} for lam, mrr in pairs]


def test_best_lambda_prefers_higher_mrr_then_the_most_balanced():
    assert best_lambda(rows((0.0, 0.5), (0.5, 0.7), (1.0, 0.6))) == 0.5
    assert best_lambda(rows((0.2, 0.7), (0.6, 0.7), (0.9, 0.7))) == 0.6
    assert best_lambda(rows((0.4, 0.7), (0.6, 0.7))) == 0.4  # equally far from 0.5: the lower


@pytest.fixture
def setup():
    chunks = recursive_chunks(make_documents(), 200, 0)
    embedder = HashEmbedder()
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embedder, persist_dir=None)
    gold = [
        {"question": "¿Qué plazo tiene el derecho de desistimiento?", "evidence": "catorce días"},
        {"question": "¿Qué es un crédito usurario?", "evidence": "usurario el crédito"},
        {"question": "¿Cómo se evalúa la solvencia?", "evidence": "evaluará la solvencia"},
        {"question": "¿Qué incluye la TAE?", "evidence": "incluye intereses y gastos"},
        {
            "question": "¿Puede el consumidor reembolsar el crédito?",
            "evidence": "reembolsar anticipadamente",
        },
        {"question": "¿Qué compensación hay por reembolso?", "evidence": "compensación equitativa"},
    ]
    resolved, _ = resolve_gold(gold, chunks)
    return chunks, embedder, bm25, dense, resolved


def test_sweep_covers_all_lambdas_and_restores_the_weight(setup):
    _, _, bm25, dense, gold = setup
    hybrid = HybridRetriever(bm25, dense, weight_bm25=0.37)
    sweep = sweep_lambda(hybrid, gold, k=5)
    assert [r["lambda"] for r in sweep] == [round(0.1 * i, 1) for i in range(11)]
    assert hybrid.weight_bm25 == 0.37
    assert all(0.0 <= r["mrr"] <= 1.0 for r in sweep)


def test_cross_validated_lambda_reports_heldout_numbers(setup):
    _, _, bm25, dense, gold = setup
    hybrid = HybridRetriever(bm25, dense, weight_bm25=0.5)
    cv = cross_validated_lambda(hybrid, gold, k=5)
    assert len(cv["lambdas_chosen"]) == 2
    assert 0.0 <= cv["mrr_heldout"] <= 1.0
    assert hybrid.weight_bm25 == 0.5


def test_a_working_retriever_finds_a_relevant_chunk_first_on_the_toy_corpus(setup):
    # Every sentence of the toy corpus is repeated in six chunks, so each question has six
    # valid chunks and Recall@5 tops out at 5/6 for a perfect retriever: MRR is the check.
    _, _, bm25, dense, gold = setup
    assert evaluate_retriever(dense, gold, 5)["mrr"] >= 0.9
    assert evaluate_retriever(bm25, gold, 5)["mrr"] >= 0.9


def test_suggest_threshold_separates_the_two_groups():
    result = suggest_threshold([0.7, 0.6, 0.65], [0.2, 0.3, 0.25])
    assert 0.3 < result["balanced"] <= 0.6
    assert result["balanced_accuracy"] == 1.0
    assert result["keep_all"] == 0.6


def test_calibrate_uses_the_best_dense_score(setup):
    _, _, _, dense, gold = setup
    report = calibrate(dense, gold, [{"question": "fútbol campeonato goles"}])
    assert report["answerable"]["min"] > report["unanswerable"]["max"]
    assert report["balanced_accuracy"] == 1.0


def test_run_strategy_end_to_end(setup, tmp_path):
    chunks, embedder, _, _, _ = setup
    gold = [
        {"question": "¿Qué plazo tiene el derecho de desistimiento?", "evidence": "catorce días"},
        {"question": "¿Qué es un crédito usurario?", "evidence": "usurario el crédito"},
        {"question": "sin evidencia en el corpus", "evidence": "texto que no existe"},
    ]
    result = run_strategy(chunks, gold, embedder, k=5, persist_dir=tmp_path)
    assert result["n_matched"] == 2 and result["n_unmatched"] == 1
    assert set(result["dense_by_instruction"]) == {"default", "minimal", "domain"}
    assert len(result["lambda_sweep"]) == 11
    assert 0.0 <= result["best_lambda"] <= 1.0
    assert {"bm25", "dense", "lambda_cv"} <= set(result)
    assert len(list(tmp_path.glob("dense_*.npz"))) == 1  # instructions share one index


def test_evaluate_generation_measures_citations_and_abstention(setup):
    chunks, _, bm25, dense, gold = setup
    hybrid = HybridRetriever(bm25, dense)
    seen_ids = {}

    def fake_generate(messages):
        prompt = messages[-1]["content"]
        if "fútbol" in prompt:
            return NO_INFO_ANSWER
        first_id = prompt.split("[")[2].split("]")[0]  # first chunk id in the prompt
        seen_ids["id"] = first_id
        return f"Respuesta apoyada en [{first_id}] y en [ffffffffff-9999]."

    pipeline = RagPipeline.from_components(
        {"bm25": bm25, "dense": dense, "hybrid": hybrid}, fake_generate, "fake"
    )
    report = evaluate_generation(pipeline, gold, [{"question": "fútbol campeonato goles"}], k=5)
    assert report["answerable"]["with_citation"] == 1.0
    assert report["answerable"]["invalid_citation_rate"] == 1.0  # the invented id is detected
    assert report["unanswerable"]["abstained"] == 1.0
    assert report["answerable"]["wrongly_abstained"] == 0.0


def test_cached_embeddings_are_computed_once_per_sentence():
    from rag.evaluate import cached

    calls = []
    base = HashEmbedder()

    def counting(texts):
        calls.append(list(texts))
        return base.embed_passages(texts)

    embed = cached(counting)
    first = embed(["uno", "dos", "uno"])
    second = embed(["dos", "tres"])
    assert first.shape == (3, 256) and second.shape == (2, 256)
    assert calls == [["uno", "dos"], ["tres"]]  # nothing re-embedded
    assert (first[0] == first[2]).all()


def test_build_strategies_creates_one_strategy_per_threshold_and_percentile():
    from rag.evaluate import build_strategies

    strategies = build_strategies(
        make_documents(), HashEmbedder(), 300, 30, [0.7], 800, percentiles=[70, 80]
    )
    assert list(strategies) == ["recursive-300", "semantic-0.7", "semantic-p70", "semantic-p80"]
    assert all(strategies[name] for name in strategies)
    # a higher percentile cuts less, so it gives fewer (or equal) chunks
    assert len(strategies["semantic-p80"]) <= len(strategies["semantic-p70"])
