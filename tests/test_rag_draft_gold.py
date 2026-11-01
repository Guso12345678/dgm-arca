"""Gold-set drafting: parsing, mechanical validation, sampling and promotion (no model)."""

import json

import pytest
from rag_helpers import make_documents

from rag.chunking import Chunk, recursive_chunks
from rag.draft_gold import (
    draft,
    lexical_overlap,
    parse_proposal,
    parse_selection,
    promote,
    sample_chunks,
    usable_chunk,
    validate_proposal,
    write_review,
)

PASSAGE = (
    "El consumidor podrá desistir del contrato de crédito en un plazo de catorce días naturales "
    "sin necesidad de indicar los motivos y sin penalización alguna para él."
)


def test_parse_proposal_accepts_fences_think_blocks_and_rejects_garbage():
    raw = '<think>hmm</think>\n```json\n{"question": "¿Plazo?", "evidence": "catorce días"}\n```'
    assert parse_proposal(raw) == {"question": "¿Plazo?", "evidence": "catorce días"}
    assert parse_proposal("no hay json") is None
    assert parse_proposal('{"question": 3, "evidence": "x"}') is None
    assert parse_proposal('{"question": "a"}') is None
    assert parse_proposal("{rota") is None


def good(**changes):
    base = {
        "question": "¿En cuánto tiempo puede arrepentirse el cliente?",
        "evidence": "plazo de catorce días naturales",
    }
    return {**base, **changes}


def test_validate_requires_literal_evidence_up_to_case_and_spacing():
    assert validate_proposal(good(), PASSAGE) is None
    assert validate_proposal(good(evidence="PLAZO  de catorce\ndías naturales"), PASSAGE) is None
    assert validate_proposal(good(evidence="plazo de quince días naturales"), PASSAGE) == (
        "evidence is not literal"
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"question": "plazo"}, "bad question"),
        (
            {"question": "¿Qué dice según el pasaje sobre el plazo del contrato?"},
            "refers to the passage",
        ),
        ({"evidence": "catorce"}, "evidence length"),
    ],
)
def test_validate_rejects_bad_proposals(changes, reason):
    assert validate_proposal(good(**changes), PASSAGE) == reason


def test_lexical_overlap_measures_word_reuse():
    copy = "¿En qué plazo de catorce días naturales puede desistir el consumidor del contrato?"
    paraphrase = "¿Cuánto tiempo tiene el cliente para arrepentirse?"
    assert lexical_overlap(copy, PASSAGE) > 0.8
    assert lexical_overlap(paraphrase, PASSAGE) < 0.3
    assert lexical_overlap("¿Qué?", PASSAGE) == 0.0


def test_usable_chunk_skips_tables_and_short_text():
    assert usable_chunk(PASSAGE * 3, 200)
    assert not usable_chunk(PASSAGE, 500)
    assert not usable_chunk("1.2 3.4 5.6 7.8 " * 60, 200)


def make_chunks():
    return recursive_chunks(make_documents(), chunk_size=400, chunk_overlap=0)


def test_sample_is_balanced_across_documents_and_reproducible():
    chunks = [Chunk(f"{s}-{i:04d}", s, f"{s}.txt", PASSAGE * 4, i) for s in "ab" for i in range(30)]
    chunks += [Chunk(f"c-{i:04d}", "c", "c.txt", PASSAGE * 4, i) for i in range(2)]
    picked = sample_chunks(chunks, 9, seed=1)
    counts = {s: sum(c.source == f"{s}.txt" for c in picked) for s in "abc"}
    # round robin a,b,c,a,b,c,a,b,a: "c" only has two chunks and runs out
    assert counts == {"a": 4, "b": 3, "c": 2}
    assert [c.id for c in sample_chunks(chunks, 9, seed=1)] == [c.id for c in picked]
    assert [c.id for c in sample_chunks(chunks, 9, seed=2)] != [c.id for c in picked]


def test_draft_keeps_valid_rejects_invalid_and_counts_reasons():
    chunks = [Chunk(f"d{i}-0000", f"d{i}", "ley.txt", PASSAGE * 4, 0) for i in range(5)]
    replies = iter(
        [
            json.dumps(good()),  # ok
            json.dumps(good(evidence="no está en el texto del pasaje")),  # not literal
            "sin json",  # unparseable
            json.dumps(good()),  # duplicate question
            json.dumps(good(question="¿Se puede desistir sin penalización alguna?")),  # ok
        ]
    )
    rows, rejected = draft(chunks, lambda prompt: next(replies), n=5)
    assert len(rows) == 2
    assert rejected == {"evidence is not literal": 1, "unparseable": 1, "duplicate": 1}
    assert {"question", "evidence", "source", "notes", "lexical_overlap", "_context"} <= set(
        rows[0]
    )


def test_high_overlap_is_flagged_in_notes():
    chunk = [Chunk("d-0000", "d", "ley.txt", PASSAGE * 4, 0)]
    copy = {
        "question": "¿Puede desistir el consumidor del contrato de crédito en catorce días?",
        "evidence": "plazo de catorce días naturales",
    }
    rows, _ = draft(chunk, lambda p: json.dumps(copy), n=1)
    assert "favours BM25" in rows[0]["notes"]


def test_parse_selection():
    assert parse_selection("1,4,7-9", 10) == [1, 4, 7, 8, 9]
    assert parse_selection("2, 2 ,3", 5) == [2, 3]
    with pytest.raises(ValueError, match="out of range"):
        parse_selection("1,11", 10)


def test_promote_appends_without_internal_fields_and_skips_repeats(tmp_path):
    rows = [
        {
            "question": "¿Uno?",
            "evidence": "e1",
            "source": "a.txt",
            "notes": "n",
            "lexical_overlap": 0.4,
            "_context": "ctx",
        },
        {
            "question": "¿Dos?",
            "evidence": "e2",
            "source": "a.txt",
            "notes": "n",
            "lexical_overlap": 0.6,
            "_context": "ctx",
        },
    ]
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"question": "¿Uno?", "evidence": "x"}) + "\n", encoding="utf-8")
    report = promote(rows, [1, 2], gold)
    assert report["added"] == 1 and report["skipped_repeats"] == 1 and report["gold_total"] == 2
    lines = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["question"] == "¿Dos?" and "_context" not in lines[-1]
    assert report["mean_lexical_overlap_of_drafted"] == 0.6


def test_review_file_lists_every_draft(tmp_path):
    rows = [
        {
            "question": "¿Uno?",
            "evidence": "e1",
            "source": "a.txt",
            "lexical_overlap": 0.4,
            "_context": "texto\ndel pasaje",
        }
    ]
    path = tmp_path / "review.md"
    write_review(rows, path)
    text = path.read_text(encoding="utf-8")
    assert "## 1. ¿Uno?" in text and "«e1»" in text and "texto del pasaje" in text
