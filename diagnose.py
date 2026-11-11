
import json
from core import load_index, retrieve

index, chunks = load_index()
eval_set = json.load(open("eval_set.json", encoding="utf-8"))

targets = [
    "without relying on consent",
    "research, archiving",
    "IndiaHandmade",
    "research infrastructure",
    "cross-disciplinary AI education"
]

for ex in eval_set:
    if any(t.lower() in ex["question"].lower() for t in targets):
        print("\n" + "=" * 80)
        print("QUESTION:", ex["question"])
        print("EXPECTED:", ex["source"], "page", ex["page"])

        results = retrieve(ex["question"], index, chunks, 8)

        for rank, r in enumerate(results, 1):
            print(f"\n{rank}. {r['source']} | page {r['page']} | score {r['score']:.4f}")
            print(r["text"][:500].replace("\n", " "))
