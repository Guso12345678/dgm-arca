"""Data indexing, step 1: collect the documents of your knowledge base.

Drop your files in ``rag/corpus/`` (PDF, Markdown, plain text, HTML) and this module
turns them into ``Document`` objects with a stable id and a ``source`` you can cite.
Keep the corpus out of git if it is big or not yours to redistribute; leave a script or
a link in ``rag/corpus/README.md`` so the index can be rebuilt (``python -m rag.fetch_corpus``).

PDFs are cleaned while they are read: official gazettes repeat a header and a footer on
every page ("BOLETÍN OFICIAL DEL ESTADO · Núm. 151 · Pág. 67844", a verification code...).
Left in, they end up inside the chunks, and a word that appears in every chunk is noise
for BM25 and a distraction for the embeddings. ``remove_repeated_lines`` drops the lines
that repeat across pages, and words split by a line break are glued back together.

Run ``uv run python -m rag.ingest`` to see what gets loaded and how long each document is.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
# A line is page boilerplate when, ignoring digits, it sits among the first or last EDGE_LINES
# lines of at least this share of the pages. Looking only at the page edges matters: a
# heading such as "Artículo 12." has the same digit-free signature as every other article
# heading and must never be mistaken for a running header.
BOILERPLATE_PAGE_SHARE = 0.5
BOILERPLATE_MIN_PAGES = 4
BOILERPLATE_MAX_CHARS = 140
EDGE_LINES = 3
# Running headers and footers are labels, not sentences: a line longer than this that ends like
# a sentence (or like a word cut by a hyphen) is content and is never dropped.
SENTENCE_LIKE_MIN_CHARS = 40
SENTENCE_ENDINGS = (".", ";", ":", ",", "-")


def _is_boilerplate_candidate(line: str) -> bool:
    stripped = line.rstrip()
    if len(stripped) > BOILERPLATE_MAX_CHARS:
        return False
    return not (len(stripped) > SENTENCE_LIKE_MIN_CHARS and stripped.endswith(SENTENCE_ENDINGS))


@dataclass
class Document:
    id: str
    source: str
    text: str
    metadata: dict | None = None


def _line_key(line: str) -> str:
    """Line signature that ignores digits, so 'Pág. 1' and 'Pág. 2' count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_indices(lines: list[str]) -> set[int]:
    """Indices of the first and last EDGE_LINES non-empty lines of a page."""
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return set(filled[:EDGE_LINES]) | set(filled[-EDGE_LINES:])


def remove_repeated_lines(pages: list[str]) -> list[str]:
    """Drop running headers, footers and page numbers: short page-edge lines that repeat.

    Pure function over the text of each page. It only acts on documents with at least
    ``BOILERPLATE_MIN_PAGES`` pages, and only on lines of at most ``BOILERPLATE_MAX_CHARS``
    that are among the first or last ``EDGE_LINES`` lines of the page.
    """
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return pages
    split = [page.splitlines() for page in pages]
    seen_in_pages: Counter[str] = Counter()
    for lines in split:
        seen_in_pages.update({_line_key(lines[i]) for i in _edge_indices(lines)})
    limit = BOILERPLATE_PAGE_SHARE * len(pages)
    repeated = {key for key, count in seen_in_pages.items() if count >= limit}
    cleaned = []
    for lines in split:
        edges = _edge_indices(lines)
        kept = [
            line
            for i, line in enumerate(lines)
            if not (i in edges and _is_boilerplate_candidate(line) and _line_key(line) in repeated)
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def join_hyphenated(text: str) -> str:
    """'con-\\ntrato' -> 'contrato': a lowercase letter on each side of a hyphen at line end."""
    return re.sub(r"(?<=[a-záéíóúñü])-\n(?=[a-záéíóúñü])", "", text)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = remove_repeated_lines([(page.extract_text() or "") for page in reader.pages])
    return join_hyphenated("\n\n".join(pages))


def _read_html(path: Path) -> str:
    """Visible text of an HTML page, one line per block element (so paragraphs survive)."""
    import html

    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(
        r"<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.I
    )
    raw = re.sub(
        r"</?(p|div|li|ul|ol|tr|br|h[1-6]|section|article|table)[^>]*>", "\n", raw, flags=re.I
    )
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines if line))


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    if path.suffix.lower() in {".html", ".htm"}:
        return _read_html(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    """Every supported file under ``corpus_dir``, recursively, as a ``Document``."""
    documents = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES | {
            ".pdf",
            ".html",
            ".htm",
        }:
            continue
        if path.name.lower() == "readme.md":  # the corpus README documents sources, it is not one
            continue
        text = read_document(path).strip()
        if not text:
            continue
        relative = str(path.relative_to(corpus_dir))
        documents.append(
            Document(
                id=hashlib.sha1(relative.encode()).hexdigest()[:10],
                source=relative,
                text=text,
                metadata={"chars": len(text), "suffix": path.suffix.lower()},
            )
        )
    return documents


if __name__ == "__main__":
    docs = load_corpus()
    if not docs:
        print(f"No documents found in {CORPUS_DIR}. Add your corpus there (see rag/README.md).")
    for doc in docs:
        print(f"{doc.id}  {doc.metadata['chars']:>8} chars  {doc.source}")
    print(f"{len(docs)} documents, {sum(d.metadata['chars'] for d in docs):,} characters in total")
