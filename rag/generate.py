"""Retrieval and generation: augment the prompt with the retrieved chunks and answer with citations.

``answer_with_rag`` is what ``POST /rag`` calls. It must:

1. retrieve ``top_k`` chunks with the requested retriever (dense, bm25 or hybrid);
2. build the augmented prompt (``RAG_PROMPT_TEMPLATE`` below is the one from class, with
   chunk ids so the model can cite them);
3. generate the answer with your model;
4. extract the citations and return everything in a ``RagResponse``.

A grader will check that every cited id exists among the returned chunks, and will read
the answers to see whether they actually rest on the retrieved text.

Two safeguards on top of the template (both measured in ``rag/evaluate.py --generation``):

* **Citations are checked.** A small model sometimes writes a chunk id it was never shown.
  ``strip_invalid_citations`` removes those ids from the text, so every ``[id]`` in the
  answer exists among the returned chunks; ``check_citations`` reports how often it happened.
* **Abstention.** When the corpus does not contain the answer, the model must say so. The
  prompt asks for it, and an optional similarity gate (``ARCA_RAG_MIN_SIMILARITY``) answers
  ``NO_INFO_ANSWER`` without calling the model when not even the best chunk is similar to
  the question. The threshold is calibrated with ``rag/evaluate.py --calibrate``.

Configuration (environment variables, so the API needs no code change):
``ARCA_RAG_CHUNKING`` (recursive | semantic), ``ARCA_RAG_SEMANTIC_THRESHOLD`` (default 0.7),
``ARCA_RAG_LAMBDA``, ``ARCA_RAG_INSTRUCTION``,
``ARCA_RAG_MODEL``, ``ARCA_RAG_EMBED_MODEL``, ``ARCA_RAG_INDEX_DIR``,
``ARCA_RAG_MIN_SIMILARITY``, ``ARCA_RAG_MAX_NEW_TOKENS``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

from api.schemas import RagResponse, RetrievedChunk
from rag.retriever import Hit

RAG_PROMPT_TEMPLATE = """Use the provided up-to-date information to answer the question.
Cite the passages you rely on by writing their id in square brackets, like [{example_id}].
If the information is not enough to answer, say so instead of guessing.

Retrieved information:
{chunks}

Question: {question}
"""

NO_INFO_ANSWER = "No hay información suficiente en los documentos para responder a esa pregunta."
SYSTEM_PROMPT = (
    "You answer questions about consumer-credit regulation using only the retrieved passages. "
    "Answer in the language of the question. Every claim must carry the id of the passage that "
    "supports it, in square brackets, copied exactly as written. If the passages do not contain "
    f"the answer, reply exactly: {NO_INFO_ANSWER}"
)

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")
# What our own ids look like (<10 hex of the document>-<4 digits>): only those are stripped.
CHUNK_ID_SHAPE = re.compile(r"^[0-9a-f]{10}-\d{4}$")
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

DEFAULT_GENERATOR = "Qwen/Qwen3-1.7B"


def format_chunks(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.id}] (from {hit.chunk.source})\n{hit.chunk.text}" for hit in hits
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = hits[0].chunk.id if hits else "chunk-id"
    return RAG_PROMPT_TEMPLATE.format(
        example_id=example_id, chunks=format_chunks(hits), question=question
    )


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def check_citations(answer: str, valid_ids: set[str]) -> dict[str, int]:
    """How many chunk-id-shaped citations the answer has, and how many of them do not exist."""
    cited = [m for m in CITATION_PATTERN.findall(answer) if CHUNK_ID_SHAPE.match(m)]
    return {"cited": len(cited), "invalid": sum(1 for m in cited if m not in valid_ids)}


def strip_invalid_citations(answer: str, valid_ids: set[str]) -> str:
    """Remove ``[id]`` markers that look like chunk ids but are not among the retrieved chunks."""

    def keep(match: re.Match) -> str:
        token = match.group(1)
        if CHUNK_ID_SHAPE.match(token) and token not in valid_ids:
            return ""
        return match.group(0)

    cleaned = CITATION_PATTERN.sub(keep, answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()


def to_response(answer: str, hits: list[Hit], retriever: str, model: str = "") -> RagResponse:
    chunks = [
        RetrievedChunk(id=h.chunk.id, source=h.chunk.source, score=h.score, text=h.chunk.text)
        for h in hits
    ]
    return RagResponse(
        answer=answer,
        chunks=chunks,
        citations=extract_citations(answer, {c.id for c in chunks}),
        retriever=retriever,
        model=model,
    )


def load_chat_generator(model_id: str, max_new_tokens: int) -> Callable[[list[dict]], str]:
    """Greedy chat generation with a transformers causal LM (thinking off). Returns a function."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def generate(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return THINK_PATTERN.sub("", text).strip()

    return generate


class RagPipeline:
    """Everything ``POST /rag`` needs, built once and reused."""

    def __init__(self) -> None:
        self.retrievers: dict[str, object] = {}
        self.dense = None
        self.generate_fn: Callable[[list[dict]], str] | None = None
        self.model_name = ""
        self.min_similarity = 0.0

    @classmethod
    def from_components(
        cls,
        retrievers: dict[str, object],
        generate_fn: Callable[[list[dict]], str],
        model_name: str = "",
        min_similarity: float = 0.0,
    ) -> RagPipeline:
        """Build a pipeline from ready parts (tests, notebooks, the evaluation)."""
        pipeline = cls()
        pipeline.retrievers = retrievers
        pipeline.dense = retrievers.get("dense")
        pipeline.generate_fn = generate_fn
        pipeline.model_name = model_name
        pipeline.min_similarity = min_similarity
        return pipeline

    def load(self) -> None:
        """Ingest -> chunk -> build the three retrievers -> load the generator model.

        The dense index is persisted under ``rag/index/`` so this is fast after the first run.
        The generator is the base Qwen3-1.7B (phase 1's LoRA adapter is for the 0.6B model);
        the name goes in ``model`` of every response.
        """
        from rag.chunking import recursive_chunks, semantic_chunks
        from rag.ingest import load_corpus
        from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

        documents = load_corpus()
        if not documents:
            raise RuntimeError(
                "rag/corpus/ is empty: run `uv run python -m rag.fetch_corpus` first "
                "(see rag/corpus/README.md)."
            )
        from rag.embed import DEFAULT_MODEL, Embedder

        embedder = Embedder(os.environ.get("ARCA_RAG_EMBED_MODEL", DEFAULT_MODEL))
        strategy = os.environ.get("ARCA_RAG_CHUNKING", "recursive")
        if strategy == "semantic":
            chunks = semantic_chunks(
                documents,
                embedder.embed_passages,
                distance_threshold=float(os.environ.get("ARCA_RAG_SEMANTIC_THRESHOLD", 0.7)),
            )
        elif strategy == "recursive":
            chunks = recursive_chunks(documents)
        else:
            raise ValueError(f"ARCA_RAG_CHUNKING must be recursive or semantic, not {strategy!r}")

        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(
            chunks,
            embedder,
            persist_dir=os.environ.get("ARCA_RAG_INDEX_DIR", "rag/index"),
            instruction=os.environ.get("ARCA_RAG_INSTRUCTION") or None,
        )
        hybrid = HybridRetriever(
            bm25, dense, weight_bm25=float(os.environ.get("ARCA_RAG_LAMBDA", 0.5))
        )
        self.retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
        self.dense = dense
        self.min_similarity = float(os.environ.get("ARCA_RAG_MIN_SIMILARITY", 0.0))
        self.model_name = os.environ.get("ARCA_RAG_MODEL", DEFAULT_GENERATOR)
        self.generate_fn = load_chat_generator(
            self.model_name, int(os.environ.get("ARCA_RAG_MAX_NEW_TOKENS", 400))
        )

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        """Dispatch to the dense, bm25 or hybrid retriever."""
        if retriever not in self.retrievers:
            raise ValueError(
                f"retriever must be one of {sorted(self.retrievers)}, not {retriever!r}"
            )
        return self.retrievers[retriever].search(question, top_k=top_k)

    def is_answerable(self, question: str) -> bool:
        """False when even the most similar chunk is below the calibrated similarity gate."""
        if self.min_similarity <= 0.0 or self.dense is None:
            return True
        return float(self.dense.scores(question).max()) >= self.min_similarity

    def generate_raw(self, question: str, hits: list[Hit]) -> str:
        """The model's answer exactly as produced (before the citations are cleaned)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ]
        return self.generate_fn(messages)

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        """Retrieve -> ``build_prompt`` -> generate -> clean the citations -> ``to_response``."""
        hits = self.retrieve(question, top_k, retriever)
        if not hits or not self.is_answerable(question):
            return to_response(NO_INFO_ANSWER, hits, retriever, self.model_name)
        raw = self.generate_raw(question, hits)
        answer = strip_invalid_citations(raw, {h.chunk.id for h in hits})
        return to_response(answer, hits, retriever, self.model_name)


_pipeline: RagPipeline | None = None


def build_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        pipeline = RagPipeline()
        pipeline.load()
        _pipeline = pipeline
    return _pipeline


def answer_with_rag(question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_pipeline().answer(question, top_k=top_k, retriever=retriever)
