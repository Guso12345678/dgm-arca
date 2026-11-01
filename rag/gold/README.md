# Gold set

`gold.jsonl` (about 50 questions, written by the team while reading the corpus) and
`unanswerable.jsonl` (questions the corpus cannot answer; used to calibrate abstention).

## Format (one JSON object per line)

```json
{"question": "¿En cuántos días puede el consumidor desistir del contrato?",
 "evidence": "catorce días naturales",
 "source": "ley_16_2011_credito_consumo.pdf",
 "notes": "paraphrased: the text says 'catorce', the question 'cuántos'"}
```

* `evidence`: a short **verbatim** passage of the corpus (20-150 characters) that contains the
  answer. Case and whitespace do not matter. `rag/evaluate.py` turns it into the ids of the
  chunks that contain it, separately for each chunking strategy, so one gold set judges both.
  Do not make it so long that a chunk boundary can cut it: that makes the question unmatched.
* `source` (optional): file name, to disambiguate text that appears in several documents.
* `relevant_chunk_ids` (optional, instead of `evidence`): explicit ids; only valid for one
  chunking, so prefer `evidence`.
* `notes` (optional): free text. Annotate the questions where two passages are equally valid.

## Check it before running anything expensive

    uv run python -m rag.evaluate --check-gold --gold rag/gold/gold.jsonl

It lists the unmatched questions (evidence not found, or split by a chunk boundary) and how many
questions have more than one valid chunk. It needs no GPU.

## What makes a gold set worth reporting

* Mix the kinds of question: a literal fact, a number or deadline, a definition, a list, a
  question that needs two passages, and **paraphrased** ones that share few words with the text
  (those are the ones where dense retrieval should beat BM25: if it does not, say so).
* Spread them over all the documents, not only the first one.
* Write the question first, from what a credit analyst would ask, and then look for the
  evidence; do not copy a sentence of the text and turn it into a question.
* `unanswerable.jsonl`: before trusting an entry, search the corpus to make sure the answer
  is not in it (the "near-domain" ones are the interesting ones; some may turn out to be
  answerable with your corpus, and then they must go).
