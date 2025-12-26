"""
Arquivo responsável por centralizar todas as configurações
globais da aplicação.

Isso facilita manutenção, troca de modelos e deploy.
"""

import streamlit as st

# ===============================
# Configurações do Ollama
# ===============================
BASE_URL = "http://localhost:11434"

MODEL_TEXT = "llama3.2:3b"
MODEL_VISION = "gemma3:12b"
EMBED_MODEL = "nomic-embed-text"

# ===============================
# Persistência
# ===============================
CHROMA_DIR = "chroma_db"
DB_URL = "sqlite:///chat_history.db"

def setup_page():
    """
    Configuração inicial da página Streamlit.
    Deve ser chamada uma única vez no app.py.
    """
    st.set_page_config(
        page_title="InkVision: Multimodal Tattoo Chatbot",
        layout="wide"
    )
