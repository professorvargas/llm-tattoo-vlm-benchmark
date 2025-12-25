import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from sqlalchemy import create_engine

# ==================================================
# LangChain / Ollama (1.x)
# ==================================================
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredImageLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ==================================================
# Configuração inicial
# ==================================================
load_dotenv(".env")

st.set_page_config(page_title="Chatbot RAG", layout="wide")
st.title("📚 Chatbot RAG (LLaMA + Gemma compatível)")

BASE_URL = "http://localhost:11434"

MODEL = "llama3.2:3b"
# MODEL = "gemma3:12b"

EMBED_MODEL = "nomic-embed-text"

user_id = st.text_input("User ID", "clayton")

# ==================================================
# Histórico persistente (SQLAlchemy)
# ==================================================
engine = create_engine("sqlite:///chat_history.db")

def get_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine,
    )

# ==================================================
# Histórico da UI
# ==================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("🧹 Nova conversa"):
    st.session_state.chat_history = []
    get_session_history(user_id).clear()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================================================
# Sidebar – Upload de documentos
# ==================================================
st.sidebar.header("📂 Base de Conhecimento (RAG)")

uploaded_files = st.sidebar.file_uploader(
    "Envie PDF, DOCX ou imagens",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# ==================================================
# Embeddings + Chroma
# ==================================================
embedding = OllamaEmbeddings(
    base_url=BASE_URL,
    model=EMBED_MODEL,
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
)

# ==================================================
# Leitura de documentos
# ==================================================
def load_documents(files):
    documents = []

    for file in files:
        suffix = file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == "docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix in ["jpg", "jpeg", "png"]:
            loader = UnstructuredImageLoader(tmp_path)
        else:
            continue

        documents.extend(loader.load())
        os.remove(tmp_path)

    return documents

if uploaded_files and st.sidebar.button("📥 Indexar documentos"):
    with st.spinner("Indexando documentos..."):
        docs = load_documents(uploaded_files)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # compatível com Gemma
            chunk_overlap=80,
        )

        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)

    st.sidebar.success("✅ Documentos indexados!")

# ==================================================
# Retriever (sempre STRING)
# ==================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

retrieve_context = RunnableLambda(
    lambda x: retriever.invoke(x["input"])
)

# ==================================================
# Histórico → texto plano (compatível Gemma)
# ==================================================
def history_to_text(history):
    if not history:
        return "Sem histórico."
    return "\n".join(
        f"{msg.type.upper()}: {msg.content}" for msg in history
    )

history_formatter = RunnableLambda(history_to_text)

# ==================================================
# Prompt ÚNICO (LLAMA + GEMMA)
# ==================================================
prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado.

Use o CONTEXTO abaixo para responder à PERGUNTA.
Se a resposta não estiver no contexto, diga claramente que não encontrou.

CONTEXTO:
{context}

HISTÓRICO:
{history}

PERGUNTA:
{input}

RESPOSTA:
"""
)

# ==================================================
# LLM (parâmetros seguros p/ ambos)
# ==================================================
llm = ChatOllama(
    base_url=BASE_URL,
    model=MODEL,
    temperature=0.2,
    num_ctx=4096,
)

# ==================================================
# Chain FINAL (RAG + Histórico)
# ==================================================
chain = (
    {
        "context": retrieve_context,
        "input": lambda x: x["input"],
        "history": lambda x: history_formatter.invoke(x["history"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ==================================================
# Streaming
# ==================================================
def chat_with_llm(session_id: str, user_input: str):
    for chunk in runnable_with_history.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk

# ==================================================
# Entrada do usuário
# ==================================================
user_prompt = st.chat_input("Pergunte algo sobre os documentos...")

if user_prompt:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(
            chat_with_llm(user_id, user_prompt)
        )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
