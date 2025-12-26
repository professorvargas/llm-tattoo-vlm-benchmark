"""
Inicialização de embeddings e VectorStore (Chroma).

Isolar isso facilita:
- Trocar Chroma por FAISS
- Trocar modelo de embedding
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config.settings import BASE_URL, EMBED_MODEL, CHROMA_DIR

def get_vectorstore():
    """
    Cria (ou carrega) o banco vetorial persistente.
    """
    embeddings = OllamaEmbeddings(
        base_url=BASE_URL,
        model=EMBED_MODEL,
    )

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
