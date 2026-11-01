"""Data indexing, step 2: chunking.

Two strategies from class. ``recursive_chunks`` is the default of most RAG systems and is
implemented for you with LangChain's splitter. ``semantic_chunks`` is yours to write: the
algorithm is in the docstring and in the slides. Compare both with ``rag/evaluate.py``:
number of chunks, average length, and Recall@k on your gold set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag.ingest import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    index: int  # position of the chunk within its document


def recursive_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[Chunk]:
    """Split by paragraphs, then lines, then sentences... until every chunk fits ``chunk_size``."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=list(separators),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            chunks.append(
                Chunk(id=f"{doc.id}-{i:04d}", doc_id=doc.id, source=doc.source, text=piece, index=i)
            )
    return chunks


# Spanish legal text is full of full stops that do not end a sentence: "art. 5", "núm. 3",
# "Artículo 12." (a heading), "a)" or "1." (list markers). Cutting there leaves fragments of
# a dozen characters whose embeddings are noise, and separates a heading from its text.
ABBREVIATIONS = {
    "art", "arts", "núm", "núms", "nº", "n.º", "apdo", "ap", "párr", "pág", "págs", "cap",
    "tít", "sec", "disp", "adic", "ed", "vol", "etc", "sr", "sra", "dr", "dra", "fig", "cfr",
    "vid", "ej", "aprox", "ud", "uds", "máx", "mín",
}  # fmt: skip
HEADING_PATTERN = re.compile(
    r"^(art[ií]culo|cap[ií]tulo|secci[oó]n|t[ií]tulo|apartado|disposici[oó]n\s+\w+)\s+[\w.º]+\.$",
    re.IGNORECASE,
)
LIST_MARKER_PATTERN = re.compile(r"^(\d{1,3}|[a-z]|[ivxlc]{1,6})[.)]$", re.IGNORECASE)


def _continues_after(piece: str) -> bool:
    """True when ``piece`` ended at an abbreviation, a heading or a list marker, not a sentence.

    A trailing "1." counts as a marker only right after ":" or ";" ("deberá constar: 1."), so a
    sentence that simply ends in a number ("el plazo es de 14.") is still cut.
    """
    words = piece.split()
    last_word = words[-1].rstrip(".").lower()
    inline_marker = (
        len(words) >= 2
        and bool(LIST_MARKER_PATTERN.match(words[-1]))
        and words[-2].endswith((":", ";"))
    )
    return (
        last_word in ABBREVIATIONS
        or bool(HEADING_PATTERN.match(piece))
        or bool(LIST_MARKER_PATTERN.match(piece))
        or inline_marker
    )


def split_sentences(text: str) -> list[str]:
    """Sentence splitter for the semantic chunker, aware of Spanish legal abbreviations.

    Splits after ``. ! ?`` and at blank lines, then glues back the pieces that were cut at an
    abbreviation ("art."), a heading ("Artículo 3.") or a list marker ("1.", "a)").
    """
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if p and p.strip()]
    sentences: list[str] = []
    glue = False
    for piece in pieces:
        if glue and sentences:
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
        glue = _continues_after(sentences[-1])
    return sentences


def sentence_distances(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance 1 - e_i · e_{i+1} between consecutive (L2-normalised) sentence vectors.

    Returns ``n - 1`` values. This is the curve of the slides: the semantic chunker cuts
    where it goes above the threshold, and ``rag/plot_distances.py`` draws it.
    """
    if len(embeddings) < 2:
        return np.zeros(0)
    return 1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Cut a single sentence longer than ``max_chars`` at word boundaries (rare: tables, lists)."""
    pieces, current = [], ""
    for word in sentence.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
        while len(current) > max_chars:  # a "word" longer than the limit (long URL)
            pieces.append(current[:max_chars])
            current = current[max_chars:]
    if current:
        pieces.append(current)
    return pieces


def group_sentences(
    sentences: list[str], distances: np.ndarray, distance_threshold: float, max_chunk_chars: int
) -> list[str]:
    """Steps 3-7 of the algorithm over precomputed distances (pure, testable without a model).

    ``distances[i]`` is the distance between sentence ``i`` and ``i + 1``. The fragments of
    a sentence that had to be cut stay together (distance 0) up to the size limit.
    """
    units: list[tuple[str, int]] = []  # (text, index of the sentence it comes from)
    for index, sentence in enumerate(sentences):
        parts = (
            _hard_split(sentence, max_chunk_chars)
            if len(sentence) > max_chunk_chars
            else [sentence]
        )
        units.extend((part, index) for part in parts)

    groups: list[list[str]] = []
    current: list[str] = []
    length = 0
    previous_sentence = -1
    for text, sentence_index in units:
        if not current:
            current, length = [text], len(text)
        else:
            gap = (
                0.0 if sentence_index == previous_sentence else float(distances[previous_sentence])
            )
            fits = length + 1 + len(text) <= max_chunk_chars
            if gap <= distance_threshold and fits:
                current.append(text)
                length += 1 + len(text)
            else:
                groups.append(current)
                current, length = [text], len(text)
        previous_sentence = sentence_index
    if current:
        groups.append(current)
    return [" ".join(group) for group in groups]


def semantic_chunks(
    documents: list[Document],
    embed_fn,
    distance_threshold: float = 0.25,
    max_chunk_chars: int = 1500,
    threshold_percentile: float | None = None,
) -> list[Chunk]:
    """Group consecutive sentences while their embeddings stay close.

    ``embed_fn(list[str]) -> array of shape (n, d)`` with L2-normalised rows (see
    ``rag/embed.py``). The algorithm from the slides:

    1. Split the document into sentences s_1 … s_n.
    2. Embed them: e_1 … e_n.
    3. Start the first chunk with s_1.
    4. For each consecutive pair, compute the cosine distance 1 - e_i · e_{i+1}.
    5. If the distance is <= threshold, append s_{i+1} to the current chunk.
    6. Otherwise close the chunk and start a new one at s_{i+1}.
    7. Also close a chunk when it would exceed ``max_chunk_chars``.

    ``threshold_percentile`` (e.g. 75) replaces the fixed threshold by the 75th percentile of
    that document's own distances, so the share of cuts is the same (25 %) in every document.
    It exists because the scale of the distances depends on the embedding model and on the
    document: with Qwen3-Embedding on legal text a fixed 0.25 would cut about 95 % of the
    boundaries (see the figure of ``rag/plot_distances.py``).

    A sentence that is longer than ``max_chunk_chars`` by itself is cut at word boundaries,
    so no chunk ever exceeds the limit. Chunk ids follow the recursive chunker's scheme,
    ``<doc_id>-<index>``: ids are only comparable within one strategy, which is why the gold
    set stores evidence text and ``rag/evaluate.py`` resolves it to ids for each strategy.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        sentences = split_sentences(doc.text)
        if not sentences:
            continue
        embeddings = np.asarray(embed_fn(sentences))
        distances = sentence_distances(embeddings)
        threshold = distance_threshold
        if threshold_percentile is not None and len(distances):
            threshold = float(np.percentile(distances, threshold_percentile))
        texts = group_sentences(sentences, distances, threshold, max_chunk_chars)
        for i, text in enumerate(texts):
            chunks.append(
                Chunk(id=f"{doc.id}-{i:04d}", doc_id=doc.id, source=doc.source, text=text, index=i)
            )
    return chunks


def describe(chunks: list[Chunk]) -> dict[str, float]:
    """Numbers for your chunking comparison table."""
    lengths = [len(c.text) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "mean_chars": sum(lengths) / max(len(lengths), 1),
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
    }
