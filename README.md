# rag-vs-vector

A benchmark that asks a simple question: **can an AI agent that browses files with simple tools match—or beat—a traditional RAG pipeline for answering questions about internal documentation?**

Inspired by the [LlamaIndex "Did filesystem tools kill vector search?" post](https://www.llamaindex.ai/blog/did-filesystem-tools-kill-vector-search), adapted for Markdown documentation from real engineering repos.

---

## What it benchmarks

Four approaches answer the same set of questions drawn from your own documentation corpus:

**RAG pipeline**
1. Split docs on Markdown headings into ~160-word chunks, each prefixed with its file path and title
2. Embed chunks with `sentence-transformers` (`all-MiniLM-L6-v2`) into a local ChromaDB store
3. At query time: retrieve candidates with both vector search and BM25, fuse with Reciprocal Rank Fusion, and group by document
4. Claude reranks the top 15 candidate documents (path + their two best-matching passages) and picks the likeliest
5. The full text of the top 4 documents goes to Claude, which answers

**Filesystem agent** — a port of LlamaIndex's [fs-explorer](https://github.com/run-llama/fs-explorer) to Claude
1. Every turn, Claude takes exactly one structured action, with a reason: a tool call (`read`, `grep`, `glob`), go one directory deeper, or stop with the answer
2. It starts from a listing of the corpus roots and is told to consult a generated `metadata.jsonl` (path, title and opening snippet of every document), as in fs-explorer's large-corpus benchmark
3. No pre-indexing—it explores your files on disk, then answers
4. Same file limit as RAG (15,000 characters per read), same set of files, and a 20-action budget after which it must answer from what it has read

**Claude Code** — the real-world baseline
1. Each question runs as a fresh headless `claude -p` session on the same model, restricted to the read-only `Read`, `Grep` and `Glob` tools
2. It sees exactly the files the other two approaches do: the benchmark copies the manifest's Markdown files into a staging folder outside any repository and runs Claude Code there with no other directory added. Its permission system denies reads anywhere else, so source code and other non-Markdown files in the corpus folders are out of reach
3. Same read budget: one `Read` returns about 15,000 characters (`CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS`, calibrated) instead of up to 2,000 lines, so long files are read in pages like the agent's
4. No personal context: `CLAUDE.md` files and auto-memory are disabled and its config directory is an empty folder, so your `~/.claude` (instructions, settings, hooks, MCP servers) is never loaded
5. Its time is the session time the CLI reports, without CLI startup; wall-clock time is recorded as well
6. Unlike the fs-explorer agent, its `Grep` searches file contents across whole folders, not one file at a time
7. Requires the [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) on your PATH; the run records its cost per question

**Claude Code + RAG search** (`claude_code_rag`) — the hybrid
1. The same Claude Code setup, plus one MCP tool, `search`, served by `approaches/rag_mcp.py`
2. `search` runs RAG's retrieval (vector search + BM25, fused, grouped by document) and returns the top 10 documents with their two best-matching passages, within the same 15,000-character budget. It makes no LLM calls: Claude Code does the reranking and answering, so all tokens show up in its own usage
3. Claude Code decides when to use `search` and when to `Grep` or `Read`
4. One server runs for the whole benchmark (the index takes ~40 s to load), started and stopped by the runner, the way RAG's warmup is excluded from its per-question time

All answers to a question are scored together, 0–10 for **correctness** and **relevance**, by an LLM judge (also Claude). The judge sees them anonymized and in a shuffled order, so it can't favor an approach by name or position. Wall-clock time is recorded for each answer.

---

## Why this matters

RAG pipelines are powerful but have real failure modes: chunking breaks context, retrieval misses the right document, and tuning the index takes effort. Agentic file search sidesteps all of that—but it's slower and hits context-window limits at scale. This benchmark lets you measure the actual tradeoff on *your own docs*.

---

## Setup

### 1. Prerequisites

- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/)

### 2. Install

```bash
git clone https://github.com/your-username/rag-vs-vector
cd rag-vs-vector
pip install -e .
```

### 3. Configure your API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Point it at your documentation

```bash
cp manifest.example.yaml manifest.yaml
# Edit manifest.yaml — add absolute paths to your doc folders
```

`manifest.yaml` is gitignored so your internal paths stay private. Each entry in the manifest is a directory of Markdown files. You can add as many as you like:

```yaml
corpus_roots:
  - path: "C:/path/to/your/wiki"
    label: wiki
    include: ["**/*.md"]
    exclude: [".attachments/**"]

  - path: "C:/path/to/your/repo/docs"
    label: my-product
    include: ["docs/**/*.md"]
    exclude: []
```

### 5. (Optional) Add a Teams channel

`corpus/teams_export.py` exports a channel's whole history as one Markdown file per thread. Graph's channel-message API needs admin consent, so it goes through the Teams web client in Edge with your own sign-in (`pip install -e .[teams]`):

```bash
python -m corpus.teams_export fetch "<channel link from Teams: … > Get link to channel>" --raw cache/teams/<name>
python -m corpus.teams_export convert cache/teams/<name> --out C:/path/to/teams-exports/<name>
```

Then add the `--out` folder to `manifest.yaml` like any other root. Re-run both steps to pick up new messages.

---

## Running the benchmark

```bash
# Step 1: Build the vector index (incremental — re-run anytime to pick up new files)
python run.py index

# Step 2: Generate questions from your corpus using Claude
python run.py generate-questions --n 20

# Step 3: Run all four approaches and score them
python run.py benchmark
# ...or a subset, e.g. skip Claude Code if you don't have the CLI:
python run.py benchmark --approaches rag,agent

# Step 4: Print the results table
python run.py report
```

Results are saved as JSON in `results/` so you can compare runs over time.

---

## Results

Run of 2026-10-15 (`results/run_20261015T202311.json`). 38 questions over 15 sources (1,509 Markdown documents, 55,674 chunks), 12 of them from a Teams escalation channel added with `corpus/teams_export.py`. Contestants on Claude Haiku 5.5, judge on Claude Sonnet 5.5.

| Approach | Correctness | Relevance | Mean time | Input tokens / question | Cost (38 questions) |
|---|---|---|---|---|---|
| RAG | **9.47** | **9.08** | **6.5 s** | **19k** | — |
| Claude Code | 9.32 | 8.68 | 14.5 s (16.5 s wall) | 71k | $0.11 |
| Claude Code + RAG search | 9.24 | 8.63 | 10.6 s (12.9 s wall) | 38k | $0.08 |
| Filesystem agent | 9.03 | 7.76 | 17.7 s | 111k | — |

By source (correctness):

| Approach | Teams threads (12) | Other docs (26) |
|---|---|---|
| RAG | 9.17 | **9.62** |
| Claude Code | 9.17 | 9.38 |
| Claude Code + RAG search | 9.00 | 9.35 |
| Filesystem agent | 9.00 | 9.04 |

**How much of this is noise.** One run per approach. The previous run (`results/run_20261015T171228.json`) had identical code for RAG, Claude Code and the agent, and each still moved by 0.3–0.4 (RAG 9.13 → 9.47, Claude Code 8.95 → 9.32, agent 8.63 → 9.03). Correctness differences under about 0.4 are not meaningful here. The time and token differences are much larger than that and are stable between runs.

**Claude Code + RAG search** scores the same as plain Claude Code, but answers 27% faster with 46% fewer input tokens and 24% lower cost. It called `search` on 36 of 38 questions, but led with it on only 4: it usually greps first and uses `search` when grep's matches don't settle it. That halves its Grep and Read calls (1.6 and 0.8 per question, against 3.2 and 1.5), and it finishes in 4.6 turns instead of 6.0. What the tool adds is fewer wasted searches, not more correct answers: plain Claude Code was already finding the right document.

One Teams question has a weak reference answer: every approach says the thread names no root cause, while the generated reference treats a participant's guess as the cause. The judge gave all four 5 this run and all three 0 in the previous one. Chat threads make weaker reference answers than documentation, because many never state a resolution.

### What changed since the first run

The first run (`results/run_20261015T121405.json`, same 38 questions) scored RAG 9.55, agent 8.76, Claude Code 8.29. Since then:

**Claude Code is held to the same conditions as the other two** (see *What it benchmarks*). It works in a staged copy of exactly the indexed documents with reads capped at about 15,000 characters, and loads no personal `CLAUDE.md` or settings. The permission boundary was checked directly: reads and globs of a source file outside the staged corpus were denied, and in the benchmark it made no tool call outside it. Two effects:

- Its correctness rose from 8.29 (to 8.95 and 9.32 in the two runs since). The old setup also *hid* files from it: its Grep honors each repository's `.gitignore`, so generated API docs and docs under `bin/` were invisible to it but not to RAG or the agent. Two of its 0 scores were exactly that. The staged copy has no `.gitignore`, so all three now see the same files.
- Its time fell from 29.6 s to 13–15 s. The figure is now the session time the CLI reports; CLI startup adds about 2 s. The rest of the drop comes from searching a smaller tree: Markdown only, no source code.

**RAG's reranker quotes each candidate's two best passages instead of one.** This was chosen on a separate dev set (75 questions from earlier question sets plus 10 new Teams questions, none sharing a source document with the test set; `benchmark/retrieval_eval.py`), not on the test questions. On the dev set:

- The source document was in the fused top 15 for 73 of 75 questions, and still 74 at top 30 or top 100. The candidate window is not the ceiling; the misses are documents neither search finds at all. Pulling more candidates from each search (150 instead of 40) made fusion worse: top 15 fell to 66.
- In the final 4 documents: 69/75 with one passage, 71/75 with two passages or with a 30-candidate window. Two passages costs fewer reranker tokens than doubling the window.

On the test set the change made no measurable difference. The source document reached RAG's context for 37 of 38 questions in every run, before and after; its score moving between 9.13 and 9.55 is answer and judge variance, not retrieval.

---

## Project structure

```
rag-vs-vector/
├── manifest.example.yaml   # template — copy to manifest.yaml
├── config.py               # settings (models, chunk size, top-k, etc.)
├── corpus/
│   ├── discover.py         # walks manifest roots, yields DocFile objects
│   └── teams_export.py     # Teams channel → one Markdown file per thread
├── pipeline/
│   └── index.py            # incremental ChromaDB indexer (SHA-256 hash cache)
├── approaches/
│   ├── rag.py              # RAG pipeline
│   ├── agent.py            # filesystem agent (fs-explorer port on Claude)
│   ├── claude_code.py      # headless Claude Code runner (with or without the RAG tool)
│   └── rag_mcp.py          # RAG retrieval as an MCP search tool, for claude_code_rag
├── benchmark/
│   ├── questions.py        # LLM-based QA pair generation (stratified by source)
│   ├── retrieval_eval.py   # RAG retrieval hit rates on a dev set (for tuning without the test set)
│   ├── runner.py           # runs each approach, records timing
│   └── judge.py            # LLM-as-judge scoring (anonymized, shuffled)
└── run.py                  # CLI entry point
```

---

## Configuration

Edit `config.py` to tune the benchmark:

| Setting | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `LLM_MODEL` | `claude-haiku-5-5` | Claude model for agent, judge, and question generation |
| `CHUNK_SIZE` | `160` | Words per chunk (the embedding model truncates at 256 tokens) |
| `CHUNK_OVERLAP` | `30` | Overlap between chunks |
| `RAG_DENSE_K` / `RAG_SPARSE_K` | `40` | Candidates from vector search / BM25 |
| `RAG_RERANK_CANDIDATES` | `15` | Documents shown to the LLM reranker |
| `RAG_RERANK_PASSAGES` | `2` | Matching passages quoted per candidate document |
| `RAG_TOP_DOCS` | `4` | Full documents passed to the answering LLM |
| `MAX_FILE_READ` | `50000` | Max chars read per file (agent + indexer) |

---

## Notes

- **No files are copied.** The index stores only embeddings and metadata (file path, label, chunk text). Your source files stay where they are.
- **Incremental indexing.** The pipeline hashes each file with SHA-256 and skips unchanged files. Re-running `index` after editing a few docs is fast.
- **Questions are generated from your docs.** Each question is designed so that the answer can only be found by reading a specific file, making the benchmark meaningful for your actual content.
- **The corpus is yours.** `manifest.yaml`, `cache/`, and `results/` are all gitignored. Nothing about your internal documentation leaves your machine unless you choose to share it.
