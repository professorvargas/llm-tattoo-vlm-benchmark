"""
Ponto de entrada da aplicação Streamlit.

Responsável por:
- Inicialização da UI
- Controle de estado do chat
- Orquestração entre RAG e Vision
"""

import streamlit as st

from config.settings import setup_page
from ui.sidebar import render_sidebar
from core.embeddings import get_vectorstore
from core.rag_chain import build_rag_chain
from services.document_loader import load_documents
from services.chat_service import chat_stream
from core.llm_vision import analyze_image_stream
from core.history import get_session_history

# ==================================================
# Configuração inicial
# ==================================================
setup_page()
st.title("💉🎨🖼️ InkVision: Multimodal Tattoo Chatbot")

user_id = st.text_input("User ID", "clayton")

# ==================================================
# Estado da sessão (UI)
# ==================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "image_docs" not in st.session_state:
    st.session_state.image_docs = []

# ==================================================
# Botão para nova conversa
# ==================================================
if st.button("🧹 Nova conversa"):
    st.session_state.chat_history = []
    st.session_state.image_docs = []
    get_session_history(user_id).clear()

# ==================================================
# Renderização do histórico do chat
# ==================================================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================================================
# VectorStore + RAG
# ==================================================
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
rag = build_rag_chain(retriever)

# ==================================================
# Sidebar – Upload e indexação
# ==================================================
uploaded_files = render_sidebar()

if uploaded_files and st.sidebar.button("📥 Indexar documentos"):
    with st.spinner("Processando documentos..."):
        text_docs, image_docs = load_documents(uploaded_files)

        if text_docs:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=80,
            )

            chunks = splitter.split_documents(text_docs)
            vectorstore.add_documents(chunks)
            st.sidebar.success("✅ Documentos textuais indexados!")

        if image_docs:
            st.session_state.image_docs.extend(image_docs)
            st.sidebar.success("🖼️ Imagens carregadas para interpretação!")

# ==================================================
# Entrada do usuário (BARRA DE CHAT)
# ==================================================
user_prompt = st.chat_input(
    "Pergunte algo sobre documentos ou desenhos..."
)

# ==================================================
# Processamento do chat
# ==================================================
if user_prompt:
    # ---- Mensagem do usuário ----
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ---- RESPOSTA DO ASSISTENTE ----
    with st.chat_message("assistant"):

        # Caso exista imagem carregada → Vision
        if st.session_state.image_docs:
            response = st.write_stream(
                analyze_image_stream(
                    st.session_state.image_docs[-1]["path"],
                    user_prompt,
                )
            )

        # Caso contrário → RAG textual
        else:
            response = st.write_stream(
                chat_stream(
                    rag,
                    user_id,
                    user_prompt,
                )
            )

    # ---- Salva resposta no histórico ----
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
