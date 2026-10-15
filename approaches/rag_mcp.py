"""
RAG's retrieval as an MCP tool, for the `claude_code_rag` candidate: Claude Code with a `search` tool
next to its Read, Grep and Glob.

`search` runs the same hybrid retrieval as the RAG pipeline (dense + BM25, fused with RRF, grouped by
document) and returns the top documents with their best-matching passages. It makes no LLM calls: Claude
Code does the reranking and answering itself, so every token it spends shows up in its own usage.
Paths are relative to Claude Code's working directory (the staged corpus), so it can Read the documents.

One server serves a whole benchmark run over local HTTP, so the ~40 s index load happens once instead of
in every fresh `claude -p` session, as RAG's own warmup is excluded from its per-question time.
`serve()` starts it and stops it when the run ends.

    python -m approaches.rag_mcp --port 8765    # run it by hand
"""
from __future__ import annotations

import argparse
import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import config

TOP_DOCS = 10
PASSAGES_PER_DOC = 2
START_TIMEOUT_S = 300

TOOL_DESCRIPTION = (
    "Search the documentation corpus by meaning and keywords (vector search plus BM25). Returns the "
    f"{TOP_DOCS} most relevant documents, best first, each with its path (relative to the current directory) "
    "and its best-matching passages. Use natural-language queries or exact terms; Read a returned path for the full document."
)


def search(query: str) -> str:
    from approaches import rag

    results = []
    budget = config.RAG_DOC_CHARS
    for rank, (_path, chunk_ids) in enumerate(rag._rank_documents(query)[:TOP_DOCS], 1):
        meta = rag._meta_by_id[chunk_ids[0]]
        passages = [rag._strip_header(rag._text_by_id.get(c, "")).strip() for c in chunk_ids[:PASSAGES_PER_DOC]]
        block = f"[{rank}] {meta['label']}/{meta['rel_path']}\n" + "\n...\n".join(p for p in passages if p)
        if len(block) > budget:
            block = block[:budget]
        results.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "\n\n".join(results) or "No matching documents."


def _build_server():
    from mcp.server.mcpserver import MCPServer
    from approaches import rag

    rag.warmup()
    server = MCPServer("rag")
    server.tool(name="search", description=TOOL_DESCRIPTION)(search)
    return server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(log_path: Path):
    """Run the MCP server in a child process for the duration of the block; yields its URL."""
    port = _free_port()
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "approaches.rag_mcp", "--port", str(port)],
        cwd=Path(__file__).resolve().parent.parent, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + START_TIMEOUT_S
        while True:  # bounded wait for the child to finish loading the index and open its port
            if proc.poll() is not None:
                raise RuntimeError(f"RAG MCP server exited during startup; see {log_path}")
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
                break
            if time.time() > deadline:
                raise RuntimeError(f"RAG MCP server did not start within {START_TIMEOUT_S}s; see {log_path}")
            time.sleep(1)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval as an MCP server (streamable HTTP)")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    server = _build_server()
    print(f"RAG MCP server ready on port {args.port}", flush=True)
    server.run("streamable-http", host="127.0.0.1", port=args.port, stateless_http=True, json_response=True)


if __name__ == "__main__":
    main()
