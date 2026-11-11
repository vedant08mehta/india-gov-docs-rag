
"""Measure retrieval quality. This is the number that goes on your resume.

Metrics:
  Hit@k : fraction of questions whose correct page appears in top-k chunks
  MRR   : mean of 1/rank of the first correct chunk
"""
import json, sys
from core import load_index, retrieve, retrieve_rerank, retrieve_hybrid

EVAL_FILE = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--file=")),
    "eval_set.json"
)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
RELAXED = "--relaxed" in sys.argv
USE_RR = "--rerank" in sys.argv
USE_HY = "--hybrid" in sys.argv


def score(eval_set, index, chunks, k):
    hits, rr = 0, 0.0

    for ex in eval_set:
        if USE_HY:
            res = retrieve_hybrid(
                ex["question"], index, chunks, k, rerank=USE_RR
            )
        else:
            res = (retrieve_rerank if USE_RR else retrieve)(
                ex["question"], index, chunks, k
            )

        ok_pages = [ex["page"]] + (
            ex.get("alt_pages", []) if RELAXED else []
        )

        rank = next(
            (i + 1 for i, r in enumerate(res)
             if r["source"] == ex["source"]
             and r["page"] in ok_pages),
            None
        )

        if rank:
            hits += 1
            rr += 1 / rank
        else:
            print("MISS:", ex["question"])

    n = len(eval_set)
    return hits / n, rr / n


if __name__ == "__main__":
    eval_set = json.load(open(EVAL_FILE))
    index, chunks = load_index()

    hit, mrr = score(eval_set, index, chunks, K)

    mode = "Relaxed" if RELAXED else "Strict"

    if USE_HY and USE_RR:
        method = "Hybrid + Reranker"
    elif USE_HY:
        method = "Hybrid"
    elif USE_RR:
        method = "Reranked"
    else:
        method = "Baseline"

    print(
        f"\n{method} | {mode} | N={len(eval_set)} | k={K} "
        f"| Hit@{K}={hit:.3f} | MRR={mrr:.3f}"
    )
