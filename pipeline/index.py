from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import config  # must precede sentence_transformers: sets HF_HUB_OFFLINE

import chromadb
from sentence_transformers import SentenceTransformer

from corpus.discover import DocFile

if TYPE_CHECKING:
    pass

_embed_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "docs"


def get_embedding_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
    return _embed_model


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma_client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_cache_db() -> sqlite3.Connection:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.INDEX_CACHE_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS file_cache "
        "(path TEXT PRIMARY KEY, sha256 TEXT, indexed_at TEXT)"
    )
    conn.commit()
    return conn


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _window(words: list[str], size: int, overlap: int) -> list[str]:
    out, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return out


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split markdown on headings, then window long sections. Each chunk keeps its heading."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            sections.append((line.lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        words = " ".join(lines).split()
        if not words and not heading:
            continue
        for piece in _window(words, chunk_size, overlap) or [""]:
            chunks.append(f"## {heading}\n{piece}" if heading else piece)
    return [c for c in chunks if c.strip()]


def _doc_header(doc: DocFile, content: str) -> str:
    title = next((l.lstrip("#").strip() for l in content.splitlines() if l.startswith("#")), Path(doc.rel_path).stem)
    return f"File: {doc.label}/{doc.rel_path}\nTitle: {title}\n"


def build_index(docs: list[DocFile], force: bool = False) -> dict:
    """
    Incrementally index docs into ChromaDB. Skips files whose SHA-256 hasn't changed.
    Returns stats: {indexed, skipped, errors, deleted}.
    """
    model = get_embedding_model()
    collection = get_collection()
    conn = _get_cache_db()
    cursor = conn.cursor()

    stats = {"indexed": 0, "skipped": 0, "errors": 0, "deleted": 0}
    current_paths = {doc.path for doc in docs}

    # Remove stale entries: files in the cache that are no longer in corpus
    cursor.execute("SELECT path FROM file_cache")
    cached_paths = {row[0] for row in cursor.fetchall()}
    stale = cached_paths - current_paths
    for stale_path in stale:
        # Delete all chunks for this file from ChromaDB
        collection.delete(where={"path": stale_path})
        cursor.execute("DELETE FROM file_cache WHERE path = ?", (stale_path,))
        stats["deleted"] += 1
    conn.commit()

    total = len(docs)
    for i, doc in enumerate(docs, 1):
        if i % 100 == 0 or i == total:
            print(f"  Indexing: {i}/{total} | indexed={stats['indexed']} skipped={stats['skipped']} errors={stats['errors']}", flush=True)

        file_hash = _sha256(doc.path)
        if not file_hash:
            stats["errors"] += 1
            continue

        if not force:
            cursor.execute("SELECT sha256 FROM file_cache WHERE path = ?", (doc.path,))
            row = cursor.fetchone()
            if row and row[0] == file_hash:
                stats["skipped"] += 1
                continue

        try:
            with open(doc.path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            print(f"  [warn] Cannot read {doc.path}: {e}")
            stats["errors"] += 1
            continue

        if not content.strip():
            stats["skipped"] += 1
            continue

        header = _doc_header(doc, content)
        chunks = [header + c for c in _chunk_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP)]
        if not chunks:
            stats["skipped"] += 1
            continue

        # Remove old chunks for this file before re-indexing
        try:
            collection.delete(where={"path": doc.path})
        except Exception:
            pass

        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"{hashlib.md5(doc.path.encode()).hexdigest()}_{j}" for j in range(len(chunks))]
        metadatas = [
            {
                "path": doc.path,
                "label": doc.label,
                "rel_path": doc.rel_path,
                "chunk_index": j,
            }
            for j in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        cursor.execute(
            "INSERT OR REPLACE INTO file_cache (path, sha256, indexed_at) VALUES (?, ?, ?)",
            (doc.path, file_hash, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        stats["indexed"] += 1

    conn.close()
    return stats
