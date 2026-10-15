# rag-vs-vector

A benchmark that asks a simple question: **can an AI agent that browses files with simple tools match—or beat—a traditional RAG pipeline for answering questions about internal documentation?**

Inspired by the [LlamaIndex "Did filesystem tools kill vector search?" post](https://www.llamaindex.ai/blog/did-filesystem-tools-kill-vector-search), adapted for Markdown documentation from real engineering repos.

---

## What it benchmarks

Three approaches answer the same set of questions drawn from your own documentation corpus:

**RAG pipeline**
1. Split docs on Markdown headings into ~160-word chunks, each prefixed with its file path and title
2. Embed chunks with `sentence-transformers` (`all-MiniLM-L6-v2`) into a local ChromaDB store
3. At query time: retrieve candidates with both vector search and BM25, fuse with Reciprocal Rank Fusion, and group by document
4. Claude reranks the top 15 candidate documents (path + best passage) and picks the likeliest
5. The full text of the top 4 documents goes to Claude, which answers

**Filesystem agent** — a port of LlamaIndex's [fs-explorer](https://github.com/run-llama/fs-explorer) to Claude
1. Every turn, Claude takes exactly one structured action, with a reason: a tool call (`read`, `grep`, `glob`), go one directory deeper, or stop with the answer
2. It starts from a listing of the corpus roots and is told to consult a generated `metadata.jsonl` (path, title and opening snippet of every document), as in fs-explorer's large-corpus benchmark
3. No pre-indexing—it explores your files on disk, then answers
4. Same file limit as RAG (15,000 characters per read), same set of files, and a 20-action budget after which it must answer from what it has read

**Claude Code** — the real-world baseline
1. Each question runs as a fresh headless `claude -p` session on the same model, restricted to the read-only `Read`, `Grep` and `Glob` tools, with the corpus folders added
2. No MCP servers, settings or hooks are loaded, and it runs from a folder outside any repository
3. Unlike the fs-explorer agent, its `Grep` searches file contents across whole folders, not one file at a time
4. Requires the [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) on your PATH; the run records its cost per question

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

# Step 3: Run all three approaches and score them
python run.py benchmark
# ...or a subset, e.g. skip Claude Code if you don't have the CLI:
python run.py benchmark --approaches rag,agent

# Step 4: Print the results table
python run.py report
```

Results are saved as JSON in `results/` so you can compare runs over time.

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
│   └── claude_code.py      # headless Claude Code runner
├── benchmark/
│   ├── questions.py        # LLM-based QA pair generation (stratified by source)
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
| `RAG_TOP_DOCS` | `4` | Full documents passed to the answering LLM |
| `MAX_FILE_READ` | `50000` | Max chars read per file (agent + indexer) |

---

## Notes

- **No files are copied.** The index stores only embeddings and metadata (file path, label, chunk text). Your source files stay where they are.
- **Incremental indexing.** The pipeline hashes each file with SHA-256 and skips unchanged files. Re-running `index` after editing a few docs is fast.
- **Questions are generated from your docs.** Each question is designed so that the answer can only be found by reading a specific file, making the benchmark meaningful for your actual content.
- **The corpus is yours.** `manifest.yaml`, `cache/`, and `results/` are all gitignored. Nothing about your internal documentation leaves your machine unless you choose to share it.
