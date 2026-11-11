
"""Interactive eval-set builder. Run: python build_eval.py [pages_per_doc]

Shows random, content-rich pages from each PDF. For each page, type ONE
question that it answers. Press Enter to skip. Saves progressively.
"""
import json, random, sys
from pathlib import Path
from pypdf import PdfReader

PER_DOC = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = Path("eval_set.json")
random.seed(42)


def good_page(text):
    t = text.strip()

    if len(t) < 900:
        return False

    head = t[:250].lower()
    if any(w in head for w in (
        "references", "bibliography", "table of contents",
        "contents", "annexure"
    )):
        return False

    letters = sum(c.isalpha() for c in t)
    if letters / max(1, len(t.replace(" ", "").replace("\n", ""))) < 0.75:
        return False

    lines = [l for l in t.split("\n") if l.strip()]
    short = sum(len(l) < 30 for l in lines)

    return short / max(1, len(lines)) < 0.4


eval_set = json.loads(OUT.read_text()) if OUT.exists() else []
done = {(e["source"], e["page"]) for e in eval_set}

for pdf in sorted(Path("data").glob("*.pdf")):
    reader = PdfReader(str(pdf))

    candidates = [
        i for i, p in enumerate(reader.pages, 1)
        if good_page(p.extract_text() or "")
        and (pdf.name, i) not in done
    ]

    selected = random.sample(candidates, min(PER_DOC, len(candidates)))

    for page_no in selected:
        print("\n" + "=" * 70)
        print(f"{pdf.name}  |  page {page_no}")
        print("=" * 70)
        print(reader.pages[page_no - 1].extract_text()[:1800])

        q = input("\nQuestion (Enter = skip, q = save and quit): ").strip()

        if q.lower() == "q":
            OUT.write_text(json.dumps(eval_set, indent=2))
            print(f"Saved {len(eval_set)} questions.")
            sys.exit()

        if q:
            eval_set.append({
                "question": q,
                "source": pdf.name,
                "page": page_no
            })
            OUT.write_text(json.dumps(eval_set, indent=2))

print(f"\nDone. {len(eval_set)} questions in {OUT}")
