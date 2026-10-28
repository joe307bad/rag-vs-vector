from __future__ import annotations

import hashlib
import json
import re
import time

from rank_bm25 import BM25Okapi

import config
from pipeline.index import get_collection, get_embedding_model

_bm25: BM25Okapi | None = None
_bm25_ids: list[str] = []
_meta_by_id: dict[str, dict] = {}
_text_by_id: dict[str, str] = {}

RERANK_PROMPT = """\
A user asked a question about internal documentation. Below are candidate documents found by search, \
each with its path and the passage that best matched. Pick the documents most likely to contain the answer. \
Pay attention to the kind of document the question refers to (a README, a run report, a blog post, a rule, a guide).

Question: {question}

{candidates}

Respond ONLY with a JSON array of up to {k} candidate numbers, most likely first, e.g. [3, 1]."""

RRF_K = 60

SYSTEM_PROMPT = "You answer questions about internal documentation using only the documents provided. " + config.ANSWER_INSTRUCTIONS


def _tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[A-Za-z0-9_.$]+", text):
        lower = word.lower().strip(".")
        if lower:
            tokens.append(lower)
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return tokens


def warmup() -> None:
    """Load the embedding model and build the in-memory BM25 index (one-time cost, excluded from per-query timing)."""
    global _bm25, _bm25_ids, _meta_by_id
    get_embedding_model()
    if _bm25 is not None:
        return
    collection = get_collection()
    ids, metas, texts = [], [], []
    page = 5_000  # one large get() exceeds SQLite's variable limit
    for offset in range(0, collection.count(), page):
        data = collection.get(include=["documents", "metadatas"], limit=page, offset=offset)
        ids += data["ids"]
        metas += data["metadatas"]
        texts += data["documents"]
    _bm25_ids = ids
    _meta_by_id = dict(zip(ids, metas))
    _text_by_id.update(zip(ids, texts))
    _bm25 = BM25Okapi([_tokenize(d) for d in texts])


def _dense_ranked(question: str) -> list[str]:
    emb = get_embedding_model().encode([question], show_progress_bar=False)[0].tolist()
    res = get_collection().query(query_embeddings=[emb], n_results=config.RAG_DENSE_K, include=["metadatas"])
    for cid, meta in zip(res["ids"][0], res["metadatas"][0]):
        _meta_by_id.setdefault(cid, meta)
    return res["ids"][0]


def _sparse_ranked(question: str) -> list[str]:
    scores = _bm25.get_scores(_tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[: config.RAG_SPARSE_K]
    return [_bm25_ids[i] for i in top if scores[i] > 0]


def _rank_documents(question: str) -> list[tuple[str, list[str]]]:
    """Return [(path, matching chunk ids, best first)] ordered by fused document score."""
    chunk_scores: dict[str, float] = {}
    for ranked in (_dense_ranked(question), _sparse_ranked(question)):
        for rank, cid in enumerate(ranked):
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    per_doc: dict[str, list[tuple[float, str]]] = {}
    for cid, score in chunk_scores.items():
        per_doc.setdefault(_meta_by_id[cid]["path"], []).append((score, cid))

    # Best chunk dominates; the next two add a small bonus. Capped so a huge document with thousands of
    # loosely matching chunks can't outrank a short document that answers the question.
    doc_score = {}
    for path, hits in per_doc.items():
        hits.sort(reverse=True)
        scores = [s for s, _ in hits]
        doc_score[path] = (scores[0] + 0.3 * sum(scores[1:3]), [cid for _, cid in hits])
    return [(p, doc_score[p][1]) for p in sorted(doc_score, key=lambda p: -doc_score[p][0])]


def _rerank(client, question: str, candidates: list[tuple[str, list[str]]], usage: dict) -> list[str]:
    """Ask the LLM which candidates most likely answer the question. Falls back to fused order."""
    listing = "\n\n".join(
        f"[{i}] {_meta_by_id[ids[0]]['label']}/{_meta_by_id[ids[0]]['rel_path']}\n{_text_by_id.get(ids[0], '')[:600]}"
        for i, (_, ids) in enumerate(candidates, 1)
    )
    fallback = [p for p, _ in candidates]
    try:
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": RERANK_PROMPT.format(question=question, candidates=listing, k=config.RAG_TOP_DOCS)}],
        )
        config.add_usage(usage, response)
        raw = next(b.text for b in response.content if hasattr(b, "text"))
        picks = json.loads(re.search(r"\[[\d,\s]*\]", raw).group(0))
        chosen = [candidates[n - 1][0] for n in picks if 1 <= n <= len(candidates)]
    except Exception:
        return fallback
    return chosen + [p for p in fallback if p not in chosen]


def _strip_header(chunk: str) -> str:
    """Chunks are stored with a 'File: ...\\nTitle: ...\\n' header; drop it when quoting the passage."""
    lines = chunk.split("\n", 2)
    return lines[2] if len(lines) == 3 and lines[0].startswith("File: ") and lines[1].startswith("Title: ") else chunk


def _document_text(path: str, matched_ids: list[str]) -> str:
    """The whole document if it fits in RAG_DOC_CHARS; otherwise its opening plus the passages that
    matched the question, so a match deep in a long document isn't cut off."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            full = f.read()
    except OSError:
        return ""
    limit = config.RAG_DOC_CHARS
    if len(full) <= limit:
        return full

    passages = [_strip_header(_text_by_id[cid]) for cid in matched_ids if cid in _text_by_id]
    budget = limit - sum(len(p) + 10 for p in passages)
    head = full[: max(2_000, budget)]
    parts = [head]
    used = len(head)
    for p in passages:
        if used + len(p) > limit:
            break
        parts.append(p)
        used += len(p)
    return "\n\n[...]\n\n".join(parts)


def answer_rag(question: str) -> dict:
    """Hybrid retrieval (dense + BM25, RRF) → top documents → full-document context → LLM."""
    if _bm25 is None:
        warmup()
    t0 = time.perf_counter()

    client = config.make_anthropic_client()
    usage: dict = {}
    candidates = _rank_documents(question)[: config.RAG_RERANK_CANDIDATES]
    matched = dict(candidates)
    ordered = _rerank(client, question, candidates, usage)

    docs: list[tuple[str, str]] = []
    seen_content: set[str] = set()
    for path in ordered:
        text = _document_text(path, matched.get(path, []))
        digest = hashlib.sha1(text.encode()).hexdigest()
        if not text.strip() or digest in seen_content:
            continue
        seen_content.add(digest)
        docs.append((path, text))
        if len(docs) >= config.RAG_TOP_DOCS:
            break

    context = "\n\n".join(f"<document path=\"{p}\">\n{t}\n</document>" for p, t in docs) or "(no documents found)"

    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    config.add_usage(usage, response)
    answer = next((b.text for b in response.content if hasattr(b, "text")), "")

    return {"answer": answer, "sources": [p for p, _ in docs], "elapsed": time.perf_counter() - t0, "usage": usage}
