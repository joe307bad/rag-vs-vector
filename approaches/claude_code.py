"""
Claude Code as a benchmark candidate: the real-world filesystem agent, run headless.

Each question is a fresh `claude -p` session with only the read-only Read, Grep and Glob tools and the
same model, API key and workspace as the other approaches. It is held to what the others get:

  - Same files. It runs inside a staged copy of exactly the documents RAG indexes and the agent sees
    (the manifest's Markdown files, nothing else from those folders), placed outside any repository.
    That folder is its working directory and no other directory is added; in headless mode Claude Code's
    permission system denies reads outside the working directory, so it cannot open source code or
    anything else on the machine.
  - Same read budget. Read's token cap is set so one read returns about RAG_DOC_CHARS of text (it otherwise
    returns up to 2,000 lines); longer files have to be read in pages with offset/limit, as the agent pages
    with start_line.
  - No personal context. CLAUDE.md files and auto-memory are disabled and the config directory is an empty
    folder, so the user's ~/.claude (CLAUDE.md, settings, hooks, MCP servers) is never loaded.
  - Same clock. `elapsed` is the session time the CLI reports, which excludes CLI startup; the wall-clock
    time including startup is kept as `wall_elapsed`.

Not `--bare`: in bare mode Claude Code offers no Grep or Glob tool (it searches through Bash),
and cross-file search is the capability this candidate is here to test.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import config

SANDBOX = Path(tempfile.gettempdir()) / "rag_vs_vector_claude"
CORPUS_DIR = SANDBOX / "corpus"
CONFIG_DIR = SANDBOX / "config"
TIMEOUT_S = 600
MODEL = "haiku"
TOOLS = "Read,Grep,Glob"
# Calibrated, not computed: Read counts its line-number prefixes and keeps a margin, so a 3,750-token cap
# returned only 8.7k chars of a prose thread and ~10k of an API doc; 6,000 returned 13.4k / 12.9k and 6,800
# returned 15.3k / 17.7k. 6,400 sits between, at about RAG_DOC_CHARS (15k).
READ_MAX_TOKENS = 6_400

TASK_PROMPT = (
    "Search the answer to the following question: '{question}' by using the Markdown (.md) documentation "
    "in the current directory (one folder per documentation source).{rag_hint}\n\n"
    "In your final response, you must report the answer to the question. "
    + config.ANSWER_INSTRUCTIONS.replace("{", "{{").replace("}", "}}")
)
RAG_TOOL = "mcp__rag__search"
RAG_HINT = " Besides Read, Grep and Glob you have a `search` tool that searches the documentation by meaning and keywords."


def stage_corpus(metadata_index: list[dict]) -> Path:
    """Mirror exactly the indexed documents into CORPUS_DIR/<label>/<rel_path>, replacing any earlier copy."""
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    for entry in metadata_index:
        dest = CORPUS_DIR / entry["label"] / entry["rel_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["path"], dest)
    if CONFIG_DIR.exists():
        shutil.rmtree(CONFIG_DIR)
    CONFIG_DIR.mkdir(parents=True)
    return CORPUS_DIR


def _env() -> dict:
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = config.ANTHROPIC_API_KEY
    env["CLAUDE_CONFIG_DIR"] = str(CONFIG_DIR)
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] = str(READ_MAX_TOKENS)
    if config.ANTHROPIC_WORKSPACE_ID:
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"anthropic-workspace-id: {config.ANTHROPIC_WORKSPACE_ID}"
    return env


def _describe_tool_use(block: dict) -> str:
    args = block.get("input", {})
    brief = args.get("file_path") or args.get("pattern") or json.dumps(args)
    if block.get("name") == "Grep" and args.get("path"):
        brief = f"{args.get('pattern')} in {args['path']}"
    return f"{block.get('name')}({str(brief)[:140]})"


def answer_claude_code(question: str, rag_url: str | None = None) -> dict:
    """Returns {answer, tool_calls, elapsed, wall_elapsed, cost_usd, num_turns}. Raises on CLI failure.
    stage_corpus() must have been called first. With rag_url (a running approaches.rag_mcp server),
    Claude Code also gets RAG retrieval as the MCP tool `search`: the claude_code_rag candidate."""
    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError("Claude Code CLI ('claude') not found on PATH")
    if not CORPUS_DIR.exists():
        raise RuntimeError("Corpus not staged; call stage_corpus() first")

    cmd = [
        exe, "-p", "--model", MODEL,
        "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        "--strict-mcp-config", "--setting-sources", "",
        "--tools", TOOLS,
    ]
    if rag_url:
        cmd += ["--mcp-config", json.dumps({"mcpServers": {"rag": {"type": "http", "url": rag_url}}}),
                "--allowedTools", RAG_TOOL]
    prompt = TASK_PROMPT.format(question=question, rag_hint=RAG_HINT if rag_url else "")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=CORPUS_DIR, env=_env(), timeout=TIMEOUT_S,
    )
    wall = time.perf_counter() - t0

    tool_calls: list[str] = []
    final: dict | None = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_calls.append(_describe_tool_use(block))
        elif event.get("type") == "result":
            final = event

    if final is None:
        raise RuntimeError(f"No result from Claude Code (exit {proc.returncode}): {proc.stderr[-500:]}")
    if final.get("is_error"):
        raise RuntimeError(f"Claude Code error: {final.get('result')}")
    denied = [str(d.get("tool_input", d))[:200] for d in final.get("permission_denials") or []]

    u = final.get("usage") or {}
    return {
        "answer": final.get("result", ""),
        "tool_calls": tool_calls,
        "permission_denials": denied,
        "elapsed": (final.get("duration_ms") or 0) / 1000 or wall,
        "wall_elapsed": wall,
        "cost_usd": final.get("total_cost_usd"),
        "num_turns": final.get("num_turns"),
        "usage": {
            "input_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0),
            "output_tokens": u.get("output_tokens") or 0,
            "cache_read_tokens": u.get("cache_read_input_tokens") or 0,
        },
    }
