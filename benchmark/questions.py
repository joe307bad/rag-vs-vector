from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import anthropic

import config
from corpus.discover import DocFile

GENERATION_PROMPT = """\
You are a documentation analyst. Read the following document excerpt and generate ONE specific question \
that can only be answered by reading this exact document. The question should require understanding \
a concrete detail, process, or fact from the document — not something that could be answered from \
general knowledge.

Also provide the reference answer (1-3 sentences, accurate and complete).

Respond ONLY with valid JSON in this exact format:
{{"question": "...", "answer": "..."}}

Document (from {label}/{rel_path}):
---
{content}
---"""


WINDOW_CHARS = 4000


def _pick_window(text: str, rng: random.Random) -> int:
    """Start offset of a random excerpt, aligned to a Markdown heading, so questions come from anywhere
    in a document rather than always its opening. Short documents are used whole."""
    if len(text) <= WINDOW_CHARS:
        return 0
    last_start = len(text) - WINDOW_CHARS // 2
    starts = [0] + [m.start() for m in re.finditer(r"^#{1,6}\s", text, re.MULTILINE) if m.start() < last_start]
    return rng.choice(starts)


def _sample_stratified(docs: list[DocFile], n: int, seed: int) -> list[DocFile]:
    """Sample n docs proportionally from each label, min 1 per label."""
    rng = random.Random(seed)
    by_label: dict[str, list[DocFile]] = {}
    for doc in docs:
        by_label.setdefault(doc.label, []).append(doc)

    num_labels = len(by_label)
    if num_labels == 0:
        return []

    # Minimum 1 per label, distribute remainder proportionally
    base = max(1, n // num_labels)
    remainder = n - base * num_labels

    sampled: list[DocFile] = []
    label_sizes = sorted(by_label.items(), key=lambda x: -len(x[1]))

    for i, (label, label_docs) in enumerate(label_sizes):
        count = base + (1 if i < remainder else 0)
        count = min(count, len(label_docs))
        chosen = rng.sample(label_docs, count)
        # Prefer docs that are not too tiny (> 200 bytes)
        chosen = [d for d in chosen if d.size_bytes > 200] or chosen
        sampled.extend(chosen)

    return sampled[:n]


def generate_questions(
    docs: list[DocFile],
    n: int = 20,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> list[dict]:
    """
    Generate n QA pairs by sampling docs and prompting claude-haiku-5-5.
    Saves to output_path (defaults to cache/questions_{n}.json).
    """
    if output_path is None:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.CACHE_DIR / f"questions_{n}.json"
    output_path = Path(output_path)

    # Filter out empty/unreadable docs
    readable = [d for d in docs if d.size_bytes > 100]
    sampled = _sample_stratified(readable, n, seed)

    print(f"  Generating {len(sampled)} questions from {len(set(d.label for d in sampled))} corpus sources...")

    client = config.make_anthropic_client()
    questions: list[dict] = []

    for i, doc in enumerate(sampled, 1):
        try:
            with open(doc.path, encoding="utf-8", errors="replace") as f:
                full = f.read()
        except OSError as e:
            print(f"  [{i}/{len(sampled)}] Skip (unreadable): {doc.rel_path} — {e}")
            continue
        window_start = _pick_window(full, random.Random(f"{seed}-{doc.path}"))
        content = full[window_start: window_start + WINDOW_CHARS]

        if not content.strip():
            print(f"  [{i}/{len(sampled)}] Skip (empty): {doc.rel_path}")
            continue

        prompt = GENERATION_PROMPT.format(
            label=doc.label,
            rel_path=doc.rel_path,
            content=content,
        )

        try:
            response = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            if not text_blocks:
                raise ValueError("No text block in response")
            raw = text_blocks[0].strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            qa = json.loads(raw)
            question_text = qa.get("question", "").strip()
            answer_text = qa.get("answer", "").strip()

            if not question_text or not answer_text:
                raise ValueError("Missing question or answer")

            entry = {
                "id": f"q{i:03d}",
                "question": question_text,
                "reference_answer": answer_text,
                "source_file": doc.path,
                "source_label": doc.label,
                "source_rel_path": doc.rel_path,
                "source_offset": window_start,
                "source_chars": len(full),
            }
            questions.append(entry)
            print(f"  [{i}/{len(sampled)}] OK: {doc.rel_path}")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  [{i}/{len(sampled)}] Parse error for {doc.rel_path}: {e}")
        except anthropic.APIError as e:
            print(f"  [{i}/{len(sampled)}] API error for {doc.rel_path}: {e}")

    output_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved {len(questions)} questions to {output_path}")
    return questions
