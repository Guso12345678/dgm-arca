"""Shared fixtures of the RAG tests: a deterministic embedder and a small credit corpus."""

import re
import zlib

import numpy as np

from rag.ingest import Document

DIM = 256


class HashEmbedder:
    """Bag-of-words hashing embedder: same interface as ``rag.embed.Embedder``, no model.

    Texts that share words get a high cosine; texts that share none are orthogonal. That is
    enough to test the retrieval logic (ranking, fusion, persistence) deterministically.
    """

    model_id = "hash-embedder"

    def __init__(self):
        self.passage_calls = 0
        self.last_instruction = None

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(DIM)
        for token in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(token.encode()) % DIM] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_passages(self, texts, batch_size=32):
        self.passage_calls += 1
        return np.stack([self._encode(t) for t in texts])

    def embed_queries(self, texts, instruction="default"):
        self.last_instruction = instruction
        return np.stack([self._encode(t) for t in texts])


TOPICS = {
    "tae": (
        "La tasa anual equivalente TAE incluye intereses y gastos "
        "y se indica en la oferta vinculante."
    ),
    "desistimiento": (
        "El consumidor puede ejercer el derecho de desistimiento "
        "en catorce días naturales sin penalización."
    ),
    "usura": (
        "Se considera usurario el crédito con interés notablemente superior al normal del dinero."
    ),
    "solvencia": (
        "El prestamista evaluará la solvencia del consumidor con información suficiente "
        "antes de conceder el crédito."
    ),
    "reembolso": (
        "El consumidor podrá reembolsar anticipadamente el crédito total o parcialmente "
        "con compensación equitativa."
    ),
}


def make_documents() -> list[Document]:
    docs = []
    for i, (key, sentence) in enumerate(TOPICS.items()):
        text = "\n\n".join(f"{sentence} Nota {key} número {n}." for n in range(6))
        docs.append(Document(id=f"doc{i:07d}", source=f"{key}.txt", text=text))
    return docs
