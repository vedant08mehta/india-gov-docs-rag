
"""Core RAG building blocks: load -> chunk -> embed -> index -> retrieve -> answer."""
import json, os, time, re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))

_client = None
def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


# ---------- 1. Load ----------
def load_pdfs(folder="data"):
    """Return a list of {source, page, text}. Page numbers are kept so answers can cite them."""
    pages = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        for i, page in enumerate(PdfReader(str(pdf)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"source": pdf.name, "page": i, "text": text})
    return pages


# ---------- 2. Chunk ----------
def chunk_pages(pages, size=800, overlap=150):
    """Sliding window over each page. Overlap stops a sentence being cut in half
    and losing its meaning. Each chunk remembers its source and page."""
    assert 0 <= overlap < size
    chunks = []
    for p in pages:
        text, start = p["text"], 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"source": p["source"], "page": p["page"], "text": piece})
            start += size - overlap
    return chunks


# ---------- 3. Embed ----------
_local = None
def embed(texts, task_type, batch=64):
    """Local embeddings (BGE-small, 384 dims): no API quota, runs on CPU."""
    global _local
    if _local is None:
        from sentence_transformers import SentenceTransformer
        _local = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if task_type == "RETRIEVAL_QUERY":
        texts = [
            "Represent this sentence for searching relevant passages: " + t
            for t in texts
        ]
    vecs = _local.encode(
        texts,
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10
    )
    return np.asarray(vecs, dtype="float32")


# ---------- 4. Index ----------
def build_index(chunks):
    vecs = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))
    return index


def load_index():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text())
    return index, chunks


# ---------- 5. Retrieve ----------
def retrieve(query, index, chunks, k=4):
    q = embed([query], "RETRIEVAL_QUERY")
    scores, ids = index.search(q, k)
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i != -1
    ]


# ---------- 5b. Rerank ----------
_rr = None
def retrieve_rerank(query, index, chunks, k=4, fetch=20):
    """Fetch `fetch` candidates with FAISS, rescore them with a cross-encoder, keep the top k."""
    global _rr
    if _rr is None:
        from sentence_transformers import CrossEncoder
        _rr = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cands = retrieve(query, index, chunks, fetch)
    scores = _rr.predict([(query, c["text"]) for c in cands])
    ranked = sorted(zip(scores, cands), key=lambda x: -x[0])[:k]
    return [{**c, "score": float(s)} for s, c in ranked]


# ---------- 5c. Hybrid BM25 + Dense Retrieval ----------
_bm25 = None

def _tok(t):
    return re.findall(r"\w+", t.lower())


def retrieve_hybrid(query, index, chunks, k=4, fetch=20, rerank=False):
    """Dense (FAISS) + keyword (BM25) candidates, fused with reciprocal rank fusion (RRF),
    optionally reranked by the cross-encoder."""
    global _bm25, _rr

    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        _bm25 = BM25Okapi([_tok(c["text"]) for c in chunks])

    key = lambda c: (c["source"], c["page"], c["text"])

    dense = retrieve(query, index, chunks, fetch)

    sparse_ids = np.argsort(_bm25.get_scores(_tok(query)))[::-1][:fetch]
    sparse = [chunks[i] for i in sparse_ids]

    fused, by_key = {}, {}

    for lst in (dense, sparse):
        for rank, c in enumerate(lst, 1):
            kk = key(c)
            by_key[kk] = {
                "source": c["source"],
                "page": c["page"],
                "text": c["text"]
            }
            fused[kk] = fused.get(kk, 0.0) + 1.0 / (60 + rank)

    order = sorted(fused, key=fused.get, reverse=True)[: 2 * fetch]
    cands = [
        {**by_key[kk], "score": fused[kk]}
        for kk in order
    ]

    if not rerank:
        return cands[:k]

    if _rr is None:
        from sentence_transformers import CrossEncoder
        _rr = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    scores = _rr.predict([(query, c["text"]) for c in cands])
    ranked = sorted(zip(scores, cands), key=lambda x: -x[0])[:k]
    return [{**c, "score": float(s)} for s, c in ranked]


# ---------- 6. Generate (grounded) ----------
PROMPT = """You answer questions using ONLY the context below.
If the context does not contain the answer, reply exactly: "I could not find this in the documents."
Cite sources like [file.pdf, p.3] after each claim.

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


def answer(query, index, chunks, k=4):
    hits = retrieve_hybrid(query, index, chunks, k, rerank=True)
    context = "\n\n".join(
        f"[{h['source']}, p.{h['page']}]\n{h['text']}"
        for h in hits
    )
    resp = client().models.generate_content(
        model=GEN_MODEL,
        contents=PROMPT.format(context=context, question=query)
    )
    return resp.text, hits
