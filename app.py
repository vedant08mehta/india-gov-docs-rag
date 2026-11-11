"""Run:  streamlit run app.py"""
import streamlit as st
from core import load_index, answer

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("Document Q&A (RAG)")

@st.cache_resource
def get_index():
    return load_index()

index, chunks = get_index()
k = st.sidebar.slider("Chunks retrieved (k)", 1, 8, 4)
q = st.text_input("Ask a question about the documents")

if q:
    with st.spinner("Searching and answering..."):
        text, hits = answer(q, index, chunks, k)
    st.write(text)
    with st.expander("Sources used"):
        for h in hits:
            st.markdown(f"**{h['source']}, p.{h['page']}** (similarity {h['score']:.2f})")
            st.caption(h["text"][:400] + "...")
