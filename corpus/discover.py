from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pathspec
import yaml


@dataclass
class DocFile:
    path: str        # absolute path (native OS separators)
    label: str       # corpus label from manifest
    rel_path: str    # path relative to corpus root
    size_bytes: int


def _resolve_path(raw: str) -> Path:
    """Resolve a path string to a Path object. Forward slashes are normalised; no URL-decoding."""
    return Path(raw)


# A nested file with a nonsense extension: only directory-level excludes (dir/**) can match it,
# so file patterns like "docs/*.md" never prune a whole directory.
_PRUNE_PROBE = "__probe__/__probe__.zz9"


def _make_spec(patterns: list[str]) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def discover_corpus(manifest_path: str | Path) -> list[DocFile]:
    """Walk all corpus roots defined in manifest.yaml, return matching DocFile objects."""
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    global_excludes: list[str] = manifest.get("global_exclude", [])
    results: list[DocFile] = []

    for entry in manifest.get("corpus_roots", []):
        root = _resolve_path(entry["path"])
        label: str = entry["label"]
        include_patterns: list[str] = entry.get("include", ["**/*.md"])
        exclude_patterns: list[str] = entry.get("exclude", []) + global_excludes

        if not root.exists():
            print(f"  [warn] Corpus root does not exist, skipping: {root}")
            continue

        include_spec = _make_spec(include_patterns)
        exclude_spec = _make_spec(exclude_patterns)
        found_before = len(results)

        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            prefix = "" if rel_dir == "." else rel_dir + "/"

            # Prune directories whose every file would be excluded (e.g. node_modules/, .git/, build/)
            # so the walk never descends into them.
            dirnames[:] = [d for d in dirnames if not exclude_spec.match_file(f"{prefix}{d}/{_PRUNE_PROBE}")]

            for name in filenames:
                rel = prefix + name
                if not include_spec.match_file(rel) or exclude_spec.match_file(rel):
                    continue
                file_path = Path(dirpath) / name
                try:
                    size = file_path.stat().st_size
                except OSError:
                    continue
                results.append(DocFile(path=str(file_path), label=label, rel_path=rel, size_bytes=size))

        ignored = _git_ignored(root, [d.rel_path for d in results[found_before:]])
        if ignored:
            results[found_before:] = [d for d in results[found_before:] if d.rel_path not in ignored]

    return results


def _git_ignored(root: Path, rel_paths: list[str]) -> set[str]:
    """Paths the repo's .gitignore excludes (generated files, build output). Leaving them out keeps the
    corpus to what's actually in the repos, and matches what git-aware tools like ripgrep search."""
    if not rel_paths:
        return set()
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-z", "--stdin"],
            input="\0".join(rel_paths), capture_output=True, text=True, encoding="utf-8",
        )
    except OSError:
        return set()
    # Exit code 1 means nothing is ignored; 128 means not a git repository.
    if proc.returncode not in (0, 1):
        return set()
    return {p for p in proc.stdout.split("\0") if p}


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return fallback


def build_metadata_index(docs: list[DocFile]) -> list[dict]:
    """
    Build a lightweight metadata index for the agent's search_metadata tool.
    Reads only the first ~500 chars of each file for title/snippet extraction.
    """
    index = []
    for doc in docs:
        try:
            with open(doc.path, encoding="utf-8", errors="replace") as f:
                head = f.read(500)
        except OSError:
            head = ""

        title = _extract_title(head, Path(doc.path).stem)
        snippet = re.sub(r"\s+", " ", head.replace("#", "").strip())[:200]
        word_count = len(head.split())

        index.append({
            "path": doc.path,
            "label": doc.label,
            "rel_path": doc.rel_path,
            "title": title,
            "snippet": snippet,
            "word_count": word_count,
            "size_bytes": doc.size_bytes,
        })

    return index
