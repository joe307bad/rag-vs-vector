import os
import sys
from pathlib import Path

import httpx2 as httpx

ROOT = Path(__file__).parent

MANIFEST_PATH = ROOT / "manifest.yaml"

if not MANIFEST_PATH.exists():
    print(
        "ERROR: manifest.yaml not found.\n"
        "Copy manifest.example.yaml to manifest.yaml and fill in your corpus paths.\n"
        f"Expected: {MANIFEST_PATH}"
    )
    sys.exit(1)

CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = CACHE_DIR / "chroma"
INDEX_CACHE_DB = CACHE_DIR / "index_cache.db"

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-haiku-5-5"      # every contestant (RAG, agent, Claude Code) and question generation
JUDGE_MODEL = "claude-sonnet-5-5"   # a different, stronger model, so no contestant grades its own answers

# Shared by all four approaches so answer style doesn't differ between them.
ANSWER_INSTRUCTIONS = (
    "Answer the question directly and completely, using exact values, commands, names and numbers as written in the source. "
    "If sources disagree, say so. If the answer is not in the documentation, say what is missing. "
    "Cite the file path(s) you used."
)

# Words per chunk. all-MiniLM-L6-v2 truncates at 256 tokens, so keep chunk + header under that.
CHUNK_SIZE = 160
CHUNK_OVERLAP = 30

RAG_DENSE_K = 40       # chunk candidates from vector search
RAG_SPARSE_K = 40      # chunk candidates from BM25
RAG_RERANK_CANDIDATES = 15  # fused candidates shown to the LLM reranker
# Matching passages quoted per candidate (1 = best passage, 600 chars; more = 400 each). Chosen on the dev set
# (cache/questions_dev.json, no overlap with the test questions): 2 passages and a 30-candidate window each
# raised source-in-final-4 from 69 to 71 of 75; 2 passages costs fewer reranker tokens.
RAG_RERANK_PASSAGES = 2
RAG_TOP_DOCS = 4       # documents whose full text is passed to the LLM
RAG_DOC_CHARS = 15_000 # per-document cap in the LLM context

if (Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_WORKSPACE_ID = os.environ.get("WORKSPACE_ID", "")

# Corporate networks with SSL inspection need verify=False on the httpx transport.
_SSL_VERIFY = os.environ.get("SSL_VERIFY", "false").lower() not in ("false", "0", "no")


def add_usage(totals: dict, response) -> None:
    """Accumulate token usage from an Anthropic response into totals."""
    u = response.usage
    totals["input_tokens"] = totals.get("input_tokens", 0) + (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    totals["output_tokens"] = totals.get("output_tokens", 0) + (u.output_tokens or 0)
    totals["cache_read_tokens"] = totals.get("cache_read_tokens", 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)


def make_anthropic_client():
    """Return an Anthropic client configured for this environment."""
    import anthropic
    extra_headers = {}
    if ANTHROPIC_WORKSPACE_ID:
        extra_headers["anthropic-workspace-id"] = ANTHROPIC_WORKSPACE_ID
    return anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        http_client=httpx.Client(verify=_SSL_VERIFY),
        default_headers=extra_headers,
    )
