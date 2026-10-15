"""
Claude Code as a benchmark candidate: the real-world filesystem agent, run headless.

Each question is a fresh `claude -p` session with only the read-only Read, Grep and Glob tools,
no MCP servers, no settings files (so no hooks or permission rules), corpus roots added as
directories, and a working folder outside any repository so no project context is picked up.
Same model, API key and workspace as the other approaches.

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

WORKSPACE = Path(tempfile.gettempdir()) / "rag_vs_vector_claude_ws"
TIMEOUT_S = 600
MODEL = "haiku"
TOOLS = "Read,Grep,Glob"

TASK_PROMPT = (
    "Search the answer to the following question: '{question}' by using the Markdown (.md) documentation "
    "in these folders:\n{roots}\n\n"
    "In your final response, you must report the answer to the question. "
    + config.ANSWER_INSTRUCTIONS.replace("{", "{{").replace("}", "}}")
)


def _env() -> dict:
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = config.ANTHROPIC_API_KEY
    if config.ANTHROPIC_WORKSPACE_ID:
        env["ANTHROPIC_CUSTOM_HEADERS"] = f"anthropic-workspace-id: {config.ANTHROPIC_WORKSPACE_ID}"
    return env


def _describe_tool_use(block: dict) -> str:
    args = block.get("input", {})
    brief = args.get("file_path") or args.get("pattern") or json.dumps(args)
    if block.get("name") == "Grep" and args.get("path"):
        brief = f"{args.get('pattern')} in {args['path']}"
    return f"{block.get('name')}({str(brief)[:140]})"


def answer_claude_code(question: str, corpus_roots: list[str]) -> dict:
    """Returns {answer, tool_calls, elapsed, cost_usd, num_turns}. Raises on CLI failure."""
    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError("Claude Code CLI ('claude') not found on PATH")
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    roots = [os.path.normpath(r) for r in corpus_roots]
    prompt = TASK_PROMPT.format(question=question, roots="\n".join(f"- {r}" for r in roots))
    cmd = [
        exe, "-p", "--model", MODEL,
        "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        "--strict-mcp-config", "--setting-sources", "",
        "--tools", TOOLS, "--allowedTools", TOOLS,
        "--add-dir", *roots,
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=WORKSPACE, env=_env(), timeout=TIMEOUT_S,
    )
    elapsed = time.perf_counter() - t0

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

    u = final.get("usage") or {}
    return {
        "answer": final.get("result", ""),
        "tool_calls": tool_calls,
        "elapsed": elapsed,
        "cli_duration_s": (final.get("duration_ms") or 0) / 1000,
        "cost_usd": final.get("total_cost_usd"),
        "num_turns": final.get("num_turns"),
        "usage": {
            "input_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0),
            "output_tokens": u.get("output_tokens") or 0,
        },
    }
