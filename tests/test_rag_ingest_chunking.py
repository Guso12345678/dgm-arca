"""PDF cleaning, HTML text and the semantic chunker (no model needed)."""

import re

import numpy as np
import pytest
from rag_helpers import HashEmbedder

from rag.chunking import (
    Chunk,
    group_sentences,
    semantic_chunks,
    sentence_distances,
    split_sentences,
)
from rag.ingest import Document, _read_html, join_hyphenated, remove_repeated_lines

# ---- ingest ------------------------------------------------------------------------------


def make_pages(n=8):
    words = [
        "intereses",
        "comisión",
        "plazo",
        "cuota",
        "garantía",
        "información",
        "contrato",
        "tasa",
    ]
    return [
        f"BOLETÍN OFICIAL DEL ESTADO\nNúm. {150 + i}  Sábado {i + 1} de junio\n"
        + "\n".join(
            f"Artículo {i}. Contenido {words[(i + j) % 8]} {words[(i * j) % 8]} línea {j}."
            for j in range(8)
        )
        + f"\nPág. {67800 + i}"
        for i in range(n)
    ]


def test_remove_repeated_lines_drops_headers_and_page_numbers_but_keeps_content():
    cleaned = remove_repeated_lines(make_pages())
    joined = "\n".join(cleaned)
    assert "BOLETÍN OFICIAL" not in joined
    assert "Pág." not in joined
    assert "Núm." not in joined
    for i in range(8):
        assert f"Artículo {i}. Contenido" in joined


def test_article_headings_are_not_mistaken_for_headers():
    # "Artículo N." has the same digit-free signature on every page where it appears
    # (here at the top of several pages) but it is content, not a running header.
    pages = [
        f"BOLETÍN OFICIAL DEL ESTADO\nArtículo {i}.\nEl prestamista informará al consumidor {i}.\n"
        f"Texto largo de la página {i}.\nOtra línea.\nMás texto.\nSigue.\nFin.\nPág. {i}"
        if i % 3 == 0
        else f"BOLETÍN OFICIAL DEL ESTADO\nTexto de continuación {i}.\nOtra.\nY otra.\n"
        f"Y otra más.\nSigue el texto.\nMás.\nFin de página.\nPág. {i}"
        for i in range(10)
    ]
    joined = "\n".join(remove_repeated_lines(pages))
    assert "BOLETÍN" not in joined
    for i in (0, 3, 6, 9):
        assert f"Artículo {i}." in joined


def test_sentence_like_lines_are_content_even_when_their_shape_repeats():
    # Every page starts with the same template line (only digits change): it ends like a
    # sentence, so it is content and must survive.
    pages = [
        f"BOLETÍN OFICIAL DEL ESTADO\nPág. {i}\n"
        f"Artículo {i}. Obligaciones del prestamista número {i} sobre la información.\n"
        + "\n".join(f"Línea distinta {i}-{j} con palabras {'x' * j}." for j in range(6))
        + f"\ncve: BOE-A-2011-{i}"
        for i in range(10)
    ]
    joined = "\n".join(remove_repeated_lines(pages))
    assert all(f"Artículo {i}. Obligaciones" in joined for i in range(10))
    assert "BOLETÍN" not in joined and "cve:" not in joined and "Pág." not in joined


def test_remove_repeated_lines_leaves_short_documents_alone():
    pages = make_pages(3)
    assert remove_repeated_lines(pages) == pages


def test_remove_repeated_lines_never_drops_long_lines():
    long_line = "Esta línea muy larga se repite pero es contenido legal " * 5
    pages = [f"{long_line}\nTexto {i}" for i in range(8)]
    assert all(long_line.strip() in page for page in remove_repeated_lines(pages))


def test_join_hyphenated_only_glues_split_words():
    assert join_hyphenated("el con-\ntrato de cré-\ndito") == "el contrato de crédito"
    assert join_hyphenated("pagos mensua-\nles") == "pagos mensuales"
    assert join_hyphenated("Directiva 2023-\n2225") == "Directiva 2023-\n2225"
    assert join_hyphenated("ex-\nPresidente") == "ex-\nPresidente"


def test_read_html_keeps_paragraphs_and_drops_navigation(tmp_path):
    page = tmp_path / "p.html"
    page.write_text(
        "<html><nav>Inicio Menú</nav><script>var x=1;</script><body><h1>Crédito</h1>"
        "<p>El crédito al consumo&nbsp;es un préstamo.</p><p>La TAE &amp; el TIN.</p>"
        "<footer>Aviso legal</footer></body></html>",
        encoding="utf-8",
    )
    text = _read_html(page)
    assert text.splitlines() == [
        "Crédito",
        "El crédito al consumo es un préstamo.",
        "La TAE & el TIN.",
    ]


# ---- chunking ----------------------------------------------------------------------------


def unit(*values):
    v = np.array(values, dtype=float)
    return v / np.linalg.norm(v)


def test_sentence_distances():
    emb = np.stack([unit(1, 0), unit(1, 0), unit(0, 1)])
    distances = sentence_distances(emb)
    assert distances == pytest.approx([0.0, 1.0])
    assert len(sentence_distances(emb[:1])) == 0


def test_group_sentences_cuts_where_distance_exceeds_threshold():
    sentences = ["a1.", "a2.", "b1.", "b2."]
    distances = np.array([0.1, 0.9, 0.1])
    assert group_sentences(sentences, distances, 0.25, 1000) == ["a1. a2.", "b1. b2."]
    assert group_sentences(sentences, distances, 0.95, 1000) == ["a1. a2. b1. b2."]


def test_group_sentences_threshold_is_inclusive():
    assert group_sentences(["x.", "y."], np.array([0.25]), 0.25, 100) == ["x. y."]


def test_group_sentences_respects_max_chars():
    sentences = ["aaaa.", "bbbb.", "cccc."]
    out = group_sentences(sentences, np.array([0.0, 0.0]), 0.25, 11)
    assert out == ["aaaa. bbbb.", "cccc."]
    assert all(len(c) <= 11 for c in out)


def test_a_sentence_longer_than_the_limit_is_cut_and_nothing_is_lost():
    long_sentence = " ".join(f"palabra{i}" for i in range(60))
    sentences = ["Corta.", long_sentence, "Final."]
    out = group_sentences(sentences, np.array([0.0, 0.0]), 0.25, 100)
    assert all(len(c) <= 100 for c in out)
    assert " ".join(out).split() == " ".join(sentences).split()


def test_group_sentences_empty_and_single():
    assert group_sentences([], np.zeros(0), 0.25, 100) == []
    assert group_sentences(["solo."], np.zeros(0), 0.25, 100) == ["solo."]


def test_semantic_chunks_cut_when_the_topic_changes():
    doc = Document(
        id="d0000000001",
        source="x.txt",
        text=(
            "La tasa anual equivalente incluye intereses. "
            "La tasa anual equivalente incluye gastos. "
            "El derecho de desistimiento dura catorce días. "
            "El derecho de desistimiento es gratuito."
        ),
    )
    chunks = semantic_chunks([doc], HashEmbedder().embed_passages, distance_threshold=0.5)
    assert len(chunks) == 2
    assert "tasa anual" in chunks[0].text and "desistimiento" in chunks[1].text
    assert [c.id for c in chunks] == ["d0000000001-0000", "d0000000001-0001"]
    assert all(isinstance(c, Chunk) and c.source == "x.txt" for c in chunks)
    assert [c.index for c in chunks] == [0, 1]


def test_semantic_chunks_text_is_complete_and_ordered():
    text = (
        "Primera frase del texto. Segunda frase distinta. Tercera frase aparte. Cuarta frase final."
    )
    doc = Document(id="d", source="s.txt", text=text)
    chunks = semantic_chunks([doc], HashEmbedder().embed_passages, 0.2, 40)
    assert " ".join(c.text for c in chunks).split() == text.split()
    assert all(len(c.text) <= 40 for c in chunks)


def test_semantic_chunks_skips_empty_documents():
    assert (
        semantic_chunks([Document(id="e", source="e.txt", text="  ")], lambda t: np.zeros((0, 2)))
        == []
    )


def test_split_sentences_basic_behaviour_is_kept():
    assert split_sentences("Uno. Dos!\n\nTres") == ["Uno.", "Dos!", "Tres"]
    assert split_sentences("") == []


def test_split_sentences_does_not_cut_at_legal_abbreviations_headings_or_markers():
    text = (
        "Artículo 3. Información precontractual. Según el art. 5 de la Ley, el consumidor "
        "recibirá la información. Véase el núm. 3 del anexo. Deberá constar: 1. El tipo. "
        "2. La TAE. a) El plazo. Fin."
    )
    assert split_sentences(text) == [
        "Artículo 3. Información precontractual.",
        "Según el art. 5 de la Ley, el consumidor recibirá la información.",
        "Véase el núm. 3 del anexo.",
        "Deberá constar: 1. El tipo.",
        "2. La TAE.",
        "a) El plazo.",
        "Fin.",
    ]


def circle_embedder(step_scale: float, seed: int = 0):
    """Sentences 'Frase número i.' are points on a circle; the distance between neighbours is
    1 - cos(angle step), so ``step_scale`` sets the scale of the distances (continuous, no ties)."""
    steps = np.random.default_rng(seed).uniform(0.2, 1.0, 400) * step_scale
    angles = np.concatenate([[0.0], np.cumsum(steps)])

    def embed(texts):
        index = [int(re.search(r"(\d+)", t).group(1)) for t in texts]
        return np.stack([[np.cos(angles[i]), np.sin(angles[i])] for i in index])

    return embed


def numbered_document(n=200):
    return Document(
        id="dn00000001",
        source="n.txt",
        text=" ".join(f"Frase número {i}." for i in range(n)),
    )


def test_percentile_threshold_cuts_the_requested_share_of_boundaries():
    doc, embed = numbered_document(), circle_embedder(0.9)
    boundaries = 199
    for percentile in (30, 70, 90):
        chunks = semantic_chunks(
            [doc], embed, max_chunk_chars=100_000, threshold_percentile=percentile
        )
        cuts = len(chunks) - 1
        assert cuts == pytest.approx(boundaries * (100 - percentile) / 100, abs=3)


def test_percentile_overrides_the_fixed_threshold():
    doc, embed = numbered_document(), circle_embedder(0.9)
    fixed = semantic_chunks([doc], embed, distance_threshold=0.0, max_chunk_chars=100_000)
    pct = semantic_chunks(
        [doc], embed, distance_threshold=0.0, max_chunk_chars=100_000, threshold_percentile=90
    )
    assert len(pct) < len(fixed)


def test_percentile_adapts_to_scale_where_a_fixed_threshold_does_not():
    # Two documents whose distances differ in scale by a factor of ~8: the same fixed threshold
    # cuts (almost) nothing in one and (almost) everything in the other; a percentile does not.
    close, far = circle_embedder(0.25, seed=1), circle_embedder(2.0, seed=2)
    doc = numbered_document()
    fixed = {
        name: len(semantic_chunks([doc], e, distance_threshold=0.25, max_chunk_chars=100_000))
        for name, e in (("close", close), ("far", far))
    }
    assert fixed["close"] < 10 and fixed["far"] > 120
    for embed in (close, far):
        chunks = semantic_chunks([doc], embed, max_chunk_chars=100_000, threshold_percentile=75)
        assert 40 <= len(chunks) <= 60  # about 25 % of the 199 boundaries
