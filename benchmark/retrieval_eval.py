"""RAG retrieval on a question set (use a dev set, not the test questions): where each source doc ranks before
reranking, and whether it reaches the final documents. No answering or judging, so it is cheap.
Env overrides: DK / SK (dense / BM25 candidates), NP (passages per rerank candidate).

    python benchmark/retrieval_eval.py cache/questions_dev.json fused
    python benchmark/retrieval_eval.py cache/questions_dev.json rerank 15,30,50
"""
import json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dotenv import load_dotenv; load_dotenv()
import config; config.ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
from approaches import rag

qs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
config.RAG_RERANK_PASSAGES = int(os.environ.get("NP", config.RAG_RERANK_PASSAGES))
print(f"passages per candidate {config.RAG_RERANK_PASSAGES}")
config.RAG_DENSE_K = int(os.environ.get("DK", config.RAG_DENSE_K))
config.RAG_SPARSE_K = int(os.environ.get("SK", config.RAG_SPARSE_K))
print(f"dense k {config.RAG_DENSE_K}, sparse k {config.RAG_SPARSE_K}")
rag.warmup()
norm = lambda p: os.path.normcase(os.path.normpath(p))
indexed = {norm(m["path"]) for m in rag._meta_by_id.values()}
dropped = [q["id"] for q in qs if norm(q["source_file"]) not in indexed]
qs = [q for q in qs if norm(q["source_file"]) in indexed]
print(f"{len(qs)} questions (dropped {dropped}: source no longer indexed)")


def fused_rank(q):
    ranked = [norm(p) for p, _ in rag._rank_documents(q["question"])]
    src = norm(q["source_file"])
    return ranked.index(src) + 1 if src in ranked else None


ranks = [fused_rank(q) for q in qs]
if mode == "fused":
    for k in (4, 15, 30, 50, 100):
        print(f"source in fused top {k:3d}: {sum(1 for r in ranks if r and r <= k)}/{len(qs)}")
    print("not retrieved at all:", sum(1 for r in ranks if r is None))
    sys.exit()

client = config.make_anthropic_client()
for k in [int(x) for x in sys.argv[3].split(",")]:
    def final_hit(q):
        cands = rag._rank_documents(q["question"])[:k]
        ordered = rag._rerank(client, q["question"], cands, {})
        return norm(q["source_file"]) in [norm(p) for p in ordered[: config.RAG_TOP_DOCS]]
    with ThreadPoolExecutor(8) as ex:
        hits = list(ex.map(final_hit, qs))
    by_label = Counter(q["source_label"] for q, h in zip(qs, hits) if not h)
    print(f"rerank window {k:3d}: source in final {config.RAG_TOP_DOCS} = {sum(hits)}/{len(qs)}   misses by source: {dict(by_label)}")
