#!/usr/bin/env python3
"""
rag-vs-vector benchmark CLI.

Commands:
  python run.py index [--force]
  python run.py generate-questions [--n N] [--seed S] [--output PATH]
  python run.py benchmark [--questions PATH] [--approaches rag,agent,claude_code,claude_code_rag]
  python run.py report [PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv  # optional — only if python-dotenv is installed

# Load .env if present
try:
    load_dotenv()
except ImportError:
    pass

# Ensure ANTHROPIC_API_KEY is present before doing anything heavy
def _require_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it in your environment or copy .env.example to .env and fill it in.")
        sys.exit(1)
    # Patch config so modules pick it up
    import config
    config.ANTHROPIC_API_KEY = key


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_result(data: dict) -> Path:
    from config import RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"run_{ts}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _print_report(data: dict):
    """Print a formatted results table."""
    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        agg = data.get("aggregate", {})

        table = Table(title="Benchmark Results", box=box.ROUNDED)
        table.add_column("Approach", style="bold cyan")
        table.add_column("Avg Correctness (0-10)", justify="right")
        table.add_column("Avg Relevance (0-10)", justify="right")
        table.add_column("Avg Time (s)", justify="right")
        table.add_column("Total Cost ($)", justify="right")
        table.add_column("N Questions", justify="right")

        for approach, stats in agg.items():
            table.add_row(
                approach,
                str(stats.get("avg_correctness", "—")),
                str(stats.get("avg_relevance", "—")),
                str(stats.get("avg_time", "—")),
                str(stats.get("total_cost_usd", "—")),
                str(stats.get("n", "—")),
            )

        console.print(table)

        # Per-question breakdown
        per_q = data.get("per_question", {})
        questions = data.get("questions", [])
        results = data.get("results", {})
        if per_q and questions:
            q_table = Table(title="Per-Question Scores (correctness/relevance, time)", box=box.SIMPLE)
            q_table.add_column("ID")
            q_table.add_column("Question (truncated)")
            for approach in agg:
                q_table.add_column(approach)

            for q in questions:
                qid = q["id"]
                cells = []
                for approach in agg:
                    s = per_q.get(qid, {}).get(approach, {})
                    t = results.get(approach, {}).get(qid, {}).get("elapsed", 0)
                    cells.append(f"{s.get('correctness', '?')}/{s.get('relevance', '?')} {t:.1f}s")
                q_table.add_row(qid, q["question"][:50], *cells)

            console.print(q_table)

    except ImportError:
        # Fallback plain-text output
        print("\n=== Benchmark Results ===")
        agg = data.get("aggregate", {})
        for approach, stats in agg.items():
            print(f"\n{approach.upper()}")
            print(f"  Avg Correctness : {stats.get('avg_correctness')}")
            print(f"  Avg Relevance   : {stats.get('avg_relevance')}")
            print(f"  Avg Time (s)    : {stats.get('avg_time')}")
            print(f"  N Questions     : {stats.get('n')}")


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_index(args):
    _require_api_key()
    from corpus.discover import discover_corpus
    from pipeline.index import build_index
    import config

    print("Discovering corpus...")
    docs = discover_corpus(config.MANIFEST_PATH)
    print(f"  Found {len(docs)} documents across {len(set(d.label for d in docs))} sources.")

    print("Building index (this may take a while on first run)...")
    stats = build_index(docs, force=args.force)
    print(f"\nIndex complete: {stats}")


def cmd_generate_questions(args):
    _require_api_key()
    from corpus.discover import discover_corpus
    from benchmark.questions import generate_questions
    import config

    print("Discovering corpus...")
    docs = discover_corpus(config.MANIFEST_PATH)
    print(f"  Found {len(docs)} documents.")

    output = Path(args.output) if args.output else None
    questions = generate_questions(docs, n=args.n, seed=args.seed, output_path=output)
    print(f"\nGenerated {len(questions)} questions.")


def cmd_benchmark(args):
    _require_api_key()
    from corpus.discover import discover_corpus, build_metadata_index
    from benchmark.runner import run_benchmark
    from benchmark.judge import judge_answers
    import config

    questions_path = Path(args.questions) if args.questions else config.CACHE_DIR / f"questions_20.json"
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        print("Run 'python run.py generate-questions' first.")
        sys.exit(1)

    questions = _load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    from benchmark.runner import APPROACHES
    approaches = [a.strip() for a in args.approaches.split(",") if a.strip() in APPROACHES]
    if not approaches:
        print(f"ERROR: --approaches must include one or more of: {', '.join(APPROACHES)}")
        sys.exit(1)

    print("Discovering corpus and building metadata index...")
    docs = discover_corpus(config.MANIFEST_PATH)
    metadata_index = build_metadata_index(docs)
    corpus_roots = list({str(Path(d.path).parent.parent) for d in docs})  # approximate roots

    # Get actual corpus root paths from manifest
    import yaml
    with open(config.MANIFEST_PATH, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    corpus_roots = [entry["path"] for entry in manifest.get("corpus_roots", [])]

    print(f"Running benchmark: {approaches} x {len(questions)} questions\n")
    results = run_benchmark(questions, approaches, metadata_index, corpus_roots=corpus_roots)

    if args.reuse_agent:
        with open(args.reuse_agent, encoding="utf-8") as f:
            results["agent"] = json.load(f)["results"]["agent"]
        approaches = sorted(set(approaches) | {"agent"})
        print(f"Reusing agent answers from {args.reuse_agent}")

    # Save raw answers immediately so a judge failure doesn't lose the benchmark data
    from config import RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    raw_path = RESULTS_DIR / f"run_{ts}_raw.json"
    raw_path.write_text(
        json.dumps({"questions": questions, "results": results, "approaches": approaches, "corpus_size": len(docs)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nRaw answers saved to {raw_path}")
    print("Scoring with LLM judge...")
    scores = judge_answers(questions, results)

    run_data = {
        "questions": questions,
        "results": results,
        "per_question": scores["per_question"],
        "aggregate": scores["aggregate"],
        "approaches": approaches,
        "corpus_size": len(docs),
    }

    out_path = _save_result(run_data)
    print(f"Final results saved to {out_path}")
    _print_report(run_data)


def cmd_judge(args):
    """Score a saved _raw.json file without re-running the benchmark."""
    _require_api_key()
    from benchmark.judge import judge_answers

    raw_path = Path(args.path) if args.path else None
    if raw_path is None:
        from config import RESULTS_DIR
        files = sorted(RESULTS_DIR.glob("run_*_raw.json"))
        if not files:
            print("No raw results found. Run 'python run.py benchmark' first.")
            sys.exit(1)
        raw_path = files[-1]
        print(f"Loading latest raw results: {raw_path}")

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    questions = raw["questions"]
    results = raw["results"]

    print(f"Scoring {len(questions)} questions with LLM judge...")
    scores = judge_answers(questions, results)

    run_data = {**raw, "per_question": scores["per_question"], "aggregate": scores["aggregate"]}
    out_path = _save_result(run_data)
    print(f"Final results saved to {out_path}")
    _print_report(run_data)


def cmd_report(args):
    path = Path(args.path) if args.path else None

    if path is None:
        from config import RESULTS_DIR
        files = sorted(f for f in RESULTS_DIR.glob("run_*.json") if "_raw" not in f.name)
        if not files:
            print("No results found. Run 'python run.py benchmark' first.")
            sys.exit(1)
        path = files[-1]
        print(f"Loading latest result: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    _print_report(data)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="RAG vs Filesystem Agent benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = subparsers.add_parser("index", help="Build/update vector index from corpus")
    p_index.add_argument("--force", action="store_true", help="Re-index all files even if unchanged")

    # generate-questions
    p_gen = subparsers.add_parser("generate-questions", help="Generate QA pairs from corpus")
    p_gen.add_argument("--n", type=int, default=20, help="Number of questions to generate (default: 20)")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed for doc sampling")
    p_gen.add_argument("--output", type=str, default=None, help="Output JSON path")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark on generated questions")
    p_bench.add_argument("--questions", type=str, default=None, help="Path to questions JSON")
    p_bench.add_argument("--approaches", type=str, default="rag,agent,claude_code,claude_code_rag", help="Comma-separated: rag,agent,claude_code,claude_code_rag")
    p_bench.add_argument("--reuse-agent", type=str, default=None, help="Take agent answers from a previous run JSON instead of re-running the agent")

    # judge
    p_judge = subparsers.add_parser("judge", help="Score a saved _raw.json file without re-running the benchmark")
    p_judge.add_argument("path", nargs="?", default=None, help="Path to _raw.json (default: latest)")

    # report
    p_report = subparsers.add_parser("report", help="Print results table from a saved run")
    p_report.add_argument("path", nargs="?", default=None, help="Path to run JSON (default: latest)")

    args = parser.parse_args()

    dispatch = {
        "index": cmd_index,
        "generate-questions": cmd_generate_questions,
        "benchmark": cmd_benchmark,
        "judge": cmd_judge,
        "report": cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
