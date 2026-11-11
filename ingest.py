
"""Run once after adding PDFs to data/: python ingest.py"""
import sys
from core import load_pdfs, chunk_pages, build_index, INDEX_DIR

size = int(sys.argv[1]) if len(sys.argv) > 1 else 800
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 150

pages = load_pdfs()
chunks = chunk_pages(pages, size, overlap)

print(f"{len(pages)} pages -> {len(chunks)} chunks (size={size}, overlap={overlap})")
build_index(chunks)
print(f"Index saved to {INDEX_DIR}/")
