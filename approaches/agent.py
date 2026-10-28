"""
Filesystem agent ported from LlamaIndex's fs-explorer (github.com/run-llama/fs-explorer), using Claude.

Same design as fs-explorer: every turn the model takes exactly one structured action —
a tool call (read / grep / glob), going one directory deeper, or stopping with a final answer —
plus the reason for it. Same prompts and flow; the task prompt is fs-explorer's large-corpus
("advanced") benchmark prompt that points the agent at a metadata.jsonl.

Deliberate differences, for this benchmark:
  - Claude Haiku instead of Gemini; the structured action is a forced-shape tool call.
  - No parse_file / check_api_key (corpus is Markdown) and no ask-human (benchmark forbids it).
  - A read returns up to RAG_DOC_CHARS (RAG's per-document budget), with paging via start_line.
  - The agent sees only corpus files, the same set RAG indexes.
  - grep returns whole matching lines (fs-explorer returns only the matched text) and says how many were cut.
  - A step budget with a forced final answer instead of a timeout that returns nothing.
  - The same answer instructions as the other approaches.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pathspec

import config

MAX_STEPS = 30
READ_CHARS = config.RAG_DOC_CHARS
LISTING_CHARS = config.RAG_DOC_CHARS
METADATA_PATH = config.CACHE_DIR / "metadata.jsonl"

SYSTEM_PROMPT = """
You are FsExplorer, an AI agent whose task is to help the user to explore the filesystem (starting from the corpus roots and, eventually, going deeper) in order to complete a given task.

Every time, you will be asked to take one of the following actions, by calling the `take_action` tool:

- Tool call - call one of the file-system tools available to you, specifically:
    + `read`: read a **text-based** file, providing its path (`file_path` parameter, a string). Long files are returned in pages; pass `start_line` (an integer) to continue from a given line
    + `grep`: grep the content of a file, providing its path and the pattern (`file_path` and `pattern` parameters, both strings). Returns the matching lines.
    + `glob`: list files within a directory that comply with a certain pattern, providing the directory path and the pattern to search for (`directory` and `pattern` parameters, both strings)
- Go deeper - go one level deeper in the filesystem, accessing a subfolder of the folder you are currently exploring (`directory` parameter)
- Stop - you have reached your goal, so you can exit, returning to the user with a final result of all the operations (`final_result` parameter)

Choose the action based on the current situation, inferred from the previous chat history.
"""

TASK_PROMPT = (
    "Search the answer to the following question: '{question}' by using one of the Markdown documents in the corpus. "
    "In order to understand what file you should be using, please consult the `{metadata}` file "
    "(one JSON object per line with the path, title and opening snippet of each document). "
    "In your final response, you must report the answer to the question. In this task, you MUST ONLY use tool calling. "
    + config.ANSWER_INSTRUCTIONS.replace("{", "{{").replace("}", "}}")
)

FINAL_ANSWER_PROMPT = (
    "You have used your action budget. Do not take any more actions. "
    "Answer the original question now, as accurately as possible, using only what you have already read. "
    "If you did not find the answer, say what you found and what is missing."
)

ACTION_TOOL = {
    "name": "take_action",
    "description": "Take the next action. Call this exactly once per turn.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Reason for taking this specific action"},
            "action": {"type": "string", "enum": ["tool_call", "go_deeper", "stop"]},
            "tool_name": {"type": "string", "enum": ["read", "grep", "glob"], "description": "For tool_call"},
            "file_path": {"type": "string", "description": "For read and grep"},
            "start_line": {"type": "integer", "description": "Optional for read: first line to return (1-based), to page through long files"},
            "pattern": {"type": "string", "description": "For grep (regex) and glob (glob pattern)"},
            "directory": {"type": "string", "description": "For glob and go_deeper"},
            "final_result": {"type": "string", "description": "For stop: the final answer"},
        },
        "required": ["reason", "action"],
    },
}


def _key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


class _Corpus:
    """In-memory view of the corpus directory tree, restricted to indexed documents."""

    def __init__(self, metadata_index: list[dict], corpus_roots: list[str]):
        self.roots = [os.path.normpath(r) for r in corpus_roots]
        self.files: dict[str, str] = {}  # key -> display path
        self.dir_files: dict[str, list[str]] = {}
        self.dir_subdirs: dict[str, set[str]] = {}
        self.dir_display: dict[str, str] = {}
        root_keys = {_key(r): r for r in self.roots}

        for entry in metadata_index:
            path = os.path.normpath(entry["path"])
            self.files[_key(path)] = path
            parent = os.path.dirname(path)
            self.dir_files.setdefault(_key(parent), []).append(path)
            child = parent
            while _key(child) not in root_keys:
                up = os.path.dirname(child)
                if up == child:
                    break
                self.dir_subdirs.setdefault(_key(up), set()).add(child)
                self.dir_display[_key(child)] = child
                child = up
            self.dir_display[_key(child)] = child

        self.files[_key(str(METADATA_PATH))] = str(METADATA_PATH)

    def describe(self, directory: str) -> str:
        k = _key(directory)
        if k not in self.dir_display:
            return f"No such directory in the corpus: {directory}"
        files = sorted(self.dir_files.get(k, []))
        subdirs = sorted(self.dir_subdirs.get(k, set()))
        text = f"Content of {self.dir_display[k]}\n"
        text += "FILES:\n- " + "\n- ".join(files) if files else "This folder has no documents directly inside it"
        text += "\nThis folder does not have any sub-folders" if not subdirs else "\nSUBFOLDERS:\n- " + "\n- ".join(subdirs)
        return _cap(text, LISTING_CHARS)

    def top_level(self) -> str:
        lines = [f"- {r}" for r in self.roots if _key(r) in self.dir_display]
        return "The corpus consists of these root folders:\n" + "\n".join(lines) + f"\n\nMetadata file:\n- {METADATA_PATH}"

    def resolve_file(self, file_path: str) -> str | None:
        return self.files.get(_key(file_path))

    def missing_file(self, file_path: str, tool: str) -> str:
        if _key(file_path) in self.dir_display:
            return f"{file_path} is a directory; `{tool}` works on a single file. Use `glob` or go deeper to find files in it."
        return f"No such file in the corpus: {file_path}"

    def read(self, file_path: str, start_line: int | None = None) -> str:
        path = self.resolve_file(file_path)
        if path is None:
            return self.missing_file(file_path, "read")
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        try:
            first = max(1, int(start_line or 1))
        except (TypeError, ValueError):
            first = 1
        out, used, last = [], 0, first - 1
        for i in range(first - 1, len(lines)):
            if used + len(lines[i]) > READ_CHARS and out:
                break
            out.append(lines[i])
            used += len(lines[i])
            last = i + 1
        text = "".join(out)
        if last < len(lines):
            text += f"\n\n... (showing lines {first}-{last} of {len(lines)}; read again with start_line={last + 1} for more)"
        return text or f"(no content at or after line {first}; the file has {len(lines)} lines)"

    def grep(self, file_path: str, pattern: str) -> str:
        path = self.resolve_file(file_path)
        if path is None:
            return self.missing_file(file_path, "grep")
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex {pattern!r}: {e}"
        with open(path, encoding="utf-8", errors="replace") as f:
            hits = [f"{i}: {line.rstrip()}" for i, line in enumerate(f, 1) if rx.search(line)]
        if not hits:
            return "No matches found"
        header = f"MATCHES for {pattern} in {path} ({len(hits)} matching lines):\n\n"
        shown, used = [], len(header)
        for h in hits:
            if used + len(h) + 1 > READ_CHARS:
                break
            shown.append(h)
            used += len(h) + 1
        text = header + "\n".join(shown)
        if len(shown) < len(hits):
            text += f"\n\n... ({len(hits) - len(shown)} more matching lines not shown; use a more specific pattern)"
        return text

    def glob(self, directory: str, pattern: str) -> str:
        k = _key(directory)
        if k not in self.dir_display:
            return f"No such directory in the corpus: {directory}"
        base = self.dir_display[k]
        spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
        matches = []
        for fk, path in self.files.items():
            if fk.startswith(k + os.sep) and spec.match_file(Path(os.path.relpath(path, base)).as_posix()):
                matches.append(path)
        if not matches:
            return "No matches found"
        return _cap(f"MATCHES for {pattern} in {base}:\n\n- " + "\n- ".join(sorted(matches)), LISTING_CHARS)


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n\n... (truncated at {limit} characters)"


def _write_metadata(metadata_index: list[dict]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for e in metadata_index:
            f.write(json.dumps({"path": e["path"], "title": e["title"], "snippet": e["snippet"]}, ensure_ascii=False) + "\n")


_corpus_cache: tuple[int, _Corpus] | None = None


def _cached(messages: list[dict]) -> list[dict]:
    """Copy of messages with a prompt-cache breakpoint on the last block, so each step re-reads the
    growing conversation from cache (as Claude Code does) instead of paying for it in full."""
    out = messages[:-1]
    last = dict(messages[-1])
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(b) if isinstance(b, dict) else b for b in content]
    if blocks and isinstance(blocks[-1], dict):
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    last["content"] = blocks
    return out + [last]


def _get_corpus(metadata_index: list[dict], corpus_roots: list[str]) -> _Corpus:
    global _corpus_cache
    if _corpus_cache is None or _corpus_cache[0] != id(metadata_index):
        _write_metadata(metadata_index)
        _corpus_cache = (id(metadata_index), _Corpus(metadata_index, corpus_roots))
    return _corpus_cache[1]


def _run_action(corpus: _Corpus, act: dict, task: str) -> tuple[str, str]:
    """Execute one action. Returns (log entry, text to send back to the model)."""
    kind = act.get("action")
    if kind == "go_deeper":
        directory = act.get("directory", "")
        listing = corpus.describe(directory)
        return f"go_deeper({directory})", (
            f"Given that the current directory ('{directory}') looks like this:\n\n```text\n{listing}\n```\n\n"
            f"And that the user is giving you this task: '{task}', what action should you take next?"
        )
    if kind == "tool_call":
        name = act.get("tool_name")
        if name == "read":
            result = corpus.read(act.get("file_path", ""), act.get("start_line"))
            log = f"read({act.get('file_path', '')}" + (f", start_line={act['start_line']})" if act.get("start_line") else ")")
        elif name == "grep":
            result = corpus.grep(act.get("file_path", ""), act.get("pattern", ""))
            log = f"grep({act.get('file_path', '')}, {act.get('pattern', '')!r})"
        elif name == "glob":
            result = corpus.glob(act.get("directory", ""), act.get("pattern", ""))
            log = f"glob({act.get('directory', '')}, {act.get('pattern', '')!r})"
        else:
            return f"invalid_tool({name})", f"Unknown tool {name!r}. Use read, grep or glob."
        return log, (
            f"Tool result for {name}:\n\n{result}\n\n"
            "Given the result from the tool call you just performed, what action should you take next?"
        )
    return f"invalid_action({kind})", f"Unknown action {kind!r}. Use tool_call, go_deeper or stop."


def answer_agent(question: str, metadata_index: list[dict], corpus_roots: list[str] | None = None) -> dict:
    """Answer a question with the fs-explorer-style agent. Returns {answer, tool_calls, forced_final, elapsed}."""
    corpus = _get_corpus(metadata_index, corpus_roots or [])
    t0 = time.perf_counter()
    client = config.make_anthropic_client()

    task = TASK_PROMPT.format(question=question, metadata=METADATA_PATH)
    messages: list[dict] = [{
        "role": "user",
        "content": (
            f"Given that the corpus looks like this:\n\n```text\n{corpus.top_level()}\n```\n\n"
            f"And that the user is giving you this task: '{task}', what action should you take first?"
        ),
    }]
    log: list[str] = []
    usage: dict = {}
    answer = ""

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[ACTION_TOOL],
            messages=_cached(messages),
        )
        config.add_usage(usage, response)
        messages.append({"role": "assistant", "content": response.content})
        calls = [b for b in response.content if b.type == "tool_use"]

        if not calls:
            # fs-explorer forces a JSON action every turn, so plain text isn't an option there either:
            # send the model back to choose an action (this uses up a step).
            log.append("no_action")
            messages.append({"role": "user", "content": "You must respond by calling `take_action`. To finish, use the `stop` action with your final answer."})
            continue

        act = dict(calls[0].input)
        if act.get("action") == "stop" and (act.get("final_result") or "").strip():
            answer = act["final_result"]
            break

        if act.get("action") == "stop":
            entry, reply = "empty_stop", "The `stop` action needs your final answer in `final_result`. Take your next action."
        else:
            entry, reply = _run_action(corpus, act, task)
        log.append(entry)
        results = [{"type": "tool_result", "tool_use_id": calls[0].id, "content": reply}]
        results += [
            {"type": "tool_result", "tool_use_id": c.id, "content": "Ignored: take exactly one action per turn.", "is_error": True}
            for c in calls[1:]
        ]
        messages.append({"role": "user", "content": results})

    forced_final = not answer.strip()
    if forced_final:
        if messages[-1]["role"] == "user":
            content = messages[-1]["content"]
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
            messages[-1]["content"] = blocks + [{"type": "text", "text": FINAL_ANSWER_PROMPT}]
        else:
            messages.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[ACTION_TOOL],
            tool_choice={"type": "none"},
            messages=_cached(messages),
        )
        config.add_usage(usage, response)
        answer = next((b.text for b in response.content if hasattr(b, "text")), "")

    return {"answer": answer, "tool_calls": log, "forced_final": forced_final, "elapsed": time.perf_counter() - t0, "usage": usage}
