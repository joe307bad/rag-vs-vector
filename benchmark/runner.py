from __future__ import annotations

import contextlib
import random
import traceback

import config

APPROACHES = ("rag", "agent", "claude_code", "claude_code_rag")


def _run_one(approach: str, question: str, metadata_index: list[dict], corpus_roots: list[str], rag_url: str | None) -> dict:
    if approach == "rag":
        from approaches.rag import answer_rag
        return answer_rag(question)
    if approach == "agent":
        from approaches.agent import answer_agent
        return answer_agent(question, metadata_index, corpus_roots=corpus_roots)
    if approach == "claude_code":
        from approaches.claude_code import answer_claude_code
        return answer_claude_code(question)
    if approach == "claude_code_rag":
        from approaches.claude_code import answer_claude_code
        return answer_claude_code(question, rag_url=rag_url)
    raise ValueError(f"Unknown approach: {approach}")


def _summary(r: dict) -> str:
    parts = [f"{r['elapsed']:.1f}s"]
    if "tool_calls" in r:
        parts.append(f"{len(r['tool_calls'])} tool calls")
    if r.get("forced_final"):
        parts.append("forced final answer")
    if r.get("cost_usd") is not None:
        parts.append(f"${r['cost_usd']:.4f}")
    return ", ".join(parts)


def run_benchmark(
    questions: list[dict],
    approaches: list[str],
    metadata_index: list[dict],
    corpus_roots: list[str] | None = None,
) -> dict:
    """Run each approach on each question. Returns {approach: {question_id: {answer, elapsed, ..., error}}}."""
    corpus_roots = corpus_roots or []
    results: dict[str, dict] = {approach: {} for approach in approaches}

    if "rag" in approaches:
        from approaches.rag import warmup
        print("  Warming up RAG (embedding model + BM25 index)...")
        warmup()

    if {"claude_code", "claude_code_rag"} & set(approaches):
        from approaches.claude_code import stage_corpus
        print(f"  Staging the indexed documents for Claude Code in {stage_corpus(metadata_index)}...")

    with contextlib.ExitStack() as stack:
        rag_url = None
        if "claude_code_rag" in approaches:
            from approaches.rag_mcp import serve
            print("  Starting the RAG MCP server for claude_code_rag (loads the index once)...")
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            rag_url = stack.enter_context(serve(config.CACHE_DIR / "rag_mcp.log"))
            print(f"  RAG MCP server at {rag_url}")

        for q in questions:
            qid = q["id"]
            print(f"\n  Question {qid}: {q['question'][:80]}...")
            # Shuffled per question (seeded) so no approach always runs first or last.
            order = approaches[:]
            random.Random(f"order-{qid}").shuffle(order)
            for approach in order:
                print(f"    Running {approach}...", end=" ", flush=True)
                try:
                    r = _run_one(approach, q["question"], metadata_index, corpus_roots, rag_url)
                    results[approach][qid] = {**r, "error": None}
                    print(f"done ({_summary(r)})")
                except Exception as e:
                    results[approach][qid] = {"answer": "", "elapsed": 0.0, "error": traceback.format_exc()}
                    print(f"ERROR: {e}")

    return results
