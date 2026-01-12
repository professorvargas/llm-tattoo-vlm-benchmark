import streamlit as st
import pymupdf4llm
import tempfile
import os
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_MATCHES = 5

# --- Helper Functions ---
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_texts(texts, embedder):
    return embedder.encode(texts, convert_to_numpy=True)

def chunk_markdown(md_chunks):
    if isinstance(md_chunks, list):
        return [chunk if isinstance(chunk, str) else chunk.get("text", str(chunk)) for chunk in md_chunks]
    return [md_chunks]

def retrieve(query, chunks, chunk_embeddings, embedder, k=TOP_K_MATCHES):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    similarities = np.dot(chunk_embeddings_norm, query_vec.T).squeeze()
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(chunks[i], similarities[i]) for i in top_indices]

def call_ollama(prompt):
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(
            OLLAMA_BASE_URL,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "No response from Ollama model.")
    except Exception as e:
        return f"Error calling Ollama API: {e}"


# --- Streamlit UI ---
st.set_page_config(page_title="PDF RAG QA", layout="centered")
st.title("📄 PDF RAG: Ask Questions About Your PDF")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Extracting and chunking PDF with PyMuPDF4LLM..."):
        try:
            md_chunks = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
            chunks = chunk_markdown(md_chunks)
            st.info(f"Extracted {len(chunks)} chunks from the PDF.")
        except Exception as e:
            st.error(f"Failed to extract PDF: {e}")
            os.remove(tmp_path)
            st.stop()

    with st.spinner("Embedding document chunks..."):
        embedder = get_embedder()
        chunk_embeddings = embed_texts(chunks, embedder)

    st.header("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if user_query:
        with st.spinner("Retrieving relevant context and generating answer..."):
            top_chunks = retrieve(user_query, chunks, chunk_embeddings, embedder)
            context = "\n\n".join([c[0] for c in top_chunks])
            prompt = f"""Answer the following question using ONLY the context below. If the answer is not in the context, say so.\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"""
            answer = call_ollama(prompt)
            st.subheader("Answer:")
            st.write(answer)
            with st.expander("Show retrieved context"):
                for i, (chunk, score) in enumerate(top_chunks):
                    st.markdown(f"**Chunk {i+1} (Score: {score:.2f})**\n\n{chunk}")
    os.remove(tmp_path)
else:
    st.info("Please upload a PDF to begin.")