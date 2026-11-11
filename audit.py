import json, sys
from core import load_index, retrieve

index, chunks = load_index()
for ex in json.load(open(sys.argv[1])):
    top = retrieve(ex["question"], index, chunks, 4)[0]
    if top["source"] == ex["source"] and top["page"] == ex["page"]:
        continue  # top result is the labelled page, so nothing to check
    label_txt = " ".join(c["text"] for c in chunks
                         if c["source"] == ex["source"] and c["page"] == ex["page"])
    print("=" * 70)
    print("Q:", ex["question"])
    print(f"LABEL  {ex['source']} p.{ex['page']}:", label_txt[:500])
    print(f"TOP-1  {top['source']} p.{top['page']}:", top["text"][:500])