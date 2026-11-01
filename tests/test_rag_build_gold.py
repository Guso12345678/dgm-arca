"""Gold building from keyword specs: the evidence must be a verbatim piece of the corpus."""

import json

from rag.build_gold import append_new, build_entries, clip_evidence, find_matches
from rag.evaluate import load_gold, normalise
from rag.ingest import Document

TEXT = (
    "El consumidor podrá desistir del contrato de crédito en un plazo de catorce días naturales "
    "sin necesidad de indicar los motivos.\n\n"
    "La compensación por reembolso anticipado no podrá superar el 1 por 100 del importe "
    "reembolsado. Otra frase corta."
)
DOCS = [Document(id="d1", source="ley_16_2011.pdf", text=TEXT)]


def test_find_matches_needs_all_keywords_and_respects_source():
    assert len(find_matches(DOCS, "ley", ["catorce", "desistir"])) == 1
    assert find_matches(DOCS, "ley", ["catorce", "reembolso"]) == []
    assert find_matches(DOCS, "directiva", ["catorce"]) == []


def test_entries_are_verbatim_and_problems_reported():
    specs = [
        {"question": "¿Plazo para desistir?", "source": "ley", "all": ["catorce"]},
        {"question": "¿Algo inexistente?", "source": "ley", "all": ["zzz"]},
    ]
    entries, problems = build_entries(DOCS, specs)
    assert len(entries) == 1 and len(problems) == 1
    assert normalise(entries[0]["evidence"]) in normalise(TEXT)


def test_long_sentence_is_clipped_to_a_verbatim_piece():
    long = "palabra " * 10 + "CLAVE " + "relleno " * 40
    piece = clip_evidence(long, ["clave"])
    assert len(piece) <= 200 and "CLAVE" in piece
    assert normalise(piece) in normalise(long)


def test_append_new_skips_questions_already_in_gold(tmp_path):
    path = tmp_path / "gold.jsonl"
    entry = {"question": "¿Plazo?", "evidence": "catorce días naturales", "source": "x"}
    assert append_new(path, [entry]) == 1
    assert append_new(path, [entry]) == 0
    assert len(load_gold(path)) == 1
    json.loads(path.read_text(encoding="utf-8").splitlines()[0])
