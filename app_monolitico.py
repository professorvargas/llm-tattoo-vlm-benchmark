import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import base64

from sqlalchemy import create_engine

# ==================================================
# LangChain / Ollama (MODERNO)
# ==================================================
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ==================================================
# Configuração inicial
# ==================================================
# load_dotenv(".env")

st.set_page_config(page_title="InkVision: Multimodal Tattoo Chatbot", layout="wide")
st.title("💉🎨🖼️ InkVision: Multimodal Tattoo Chatbot")

BASE_URL = "http://localhost:11434"

MODEL_TEXT = "llama3.2:3b"
# MODEL_TEXT = "gemma3:12b"

MODEL_VISION = "gemma3:12b"
#MODEL_VISION = "llama3.2-vision"
EMBED_MODEL = "nomic-embed-text"

user_id = st.text_input("User ID", "clayton")

# ==================================================
# Histórico persistente
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

if "image_docs" not in st.session_state:
    st.session_state.image_docs = []

if st.button("🧹 Nova conversa"):
    st.session_state.chat_history = []
    st.session_state.image_docs = []
    get_session_history(user_id).clear()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================================================
# Sidebar – Upload
# ==================================================
st.sidebar.header("📂 Base de Conhecimento")

uploaded_files = st.sidebar.file_uploader(
    "Envie PDF, DOCX ou IMAGENS (desenhos)",
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
# Utilidades
# ==================================================
def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ==================================================
# Carregamento de documentos
# ==================================================
def load_documents(files):
    text_docs = []
    image_docs = []

    for file in files:
        suffix = file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            text_docs.extend(PyPDFLoader(tmp_path).load())

        elif suffix == "docx":
            text_docs.extend(Docx2txtLoader(tmp_path).load())

        elif suffix in ["jpg", "jpeg", "png"]:
            image_docs.append(
                {
                    "path": tmp_path,
                    "name": file.name,
                }
            )

        else:
            os.remove(tmp_path)

    return text_docs, image_docs

# ==================================================
# Indexação
# ==================================================
if uploaded_files and st.sidebar.button("📥 Indexar documentos"):
    with st.spinner("Processando documentos..."):
        text_docs, image_docs = load_documents(uploaded_files)

        if text_docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=80,
            )
            chunks = splitter.split_documents(text_docs)
            vectorstore.add_documents(chunks)
            st.sidebar.success("✅ Documentos textuais indexados!")
        else:
            st.sidebar.warning("⚠️ Nenhum texto encontrado para indexação.")

        if image_docs:
            st.session_state.image_docs.extend(image_docs)
            st.sidebar.success("🖼️ Imagens carregadas para interpretação!")

# ==================================================
# Retriever
# ==================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

retrieve_context = RunnableLambda(
    lambda x: retriever.invoke(x["input"])
)

def history_to_text(history):
    if not history:
        return "Sem histórico."
    return "\n".join(f"{m.type.upper()}: {m.content}" for m in history)

history_formatter = RunnableLambda(history_to_text)

# ==================================================
# Prompt textual
# ==================================================
prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado.

Use APENAS o CONTEXTO para responder.
Se não encontrar a resposta, diga claramente que não encontrou.

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
# LLM TEXTO
# ==================================================
llm_text = ChatOllama(
    # URL do servidor Ollama
    base_url=BASE_URL,

    # Modelo de linguagem (LLaMA ou Gemma)
    model=MODEL_TEXT,

    # ------------------------------------------------------------------
    # PARÂMETROS DE CRIATIVIDADE / ALEATORIEDADE
    # ------------------------------------------------------------------

    # Controla aleatoriedade (0.0 = determinístico)
    temperature=0.0,

    # Nucleus sampling
    top_p=0.9,

    # Top-K sampling
    top_k=40,

    # ------------------------------------------------------------------
    # CONTROLE DE REPETIÇÃO
    # ------------------------------------------------------------------

    # Penaliza repetições
    repeat_penalty=1.15,

    # Introdução de novos conceitos
    presence_penalty=0.0,

    # Penalização por frequência
    frequency_penalty=0.0,

    # ------------------------------------------------------------------
    # PARÂMETROS ESTRUTURAIS
    # ------------------------------------------------------------------

    # Janela de contexto
    num_ctx=4096,

    # Máx. tokens gerados
    num_predict=512,
)

# ==================================================
# LLM VISION (STREAMING)
# ==================================================
# llm_vision = ChatOllama(
#     base_url=BASE_URL,
#     model=MODEL_VISION,
#     temperature=0.2,
# )
llm_vision = ChatOllama(
    base_url=BASE_URL,
    model=MODEL_VISION,

    # ------------------------------------------------------------------
    # PARÂMETROS DE CRIATIVIDADE / GERAÇÃO (VISION)
    # ------------------------------------------------------------------

    # Controla o grau de interpretação subjetiva da imagem
    # 0.0 → descrição literal / técnica
    # 0.2–0.4 → análise clara e objetiva (RECOMENDADO)
    # 0.6+ → interpretação criativa / especulativa
    temperature=0.0,

    # Nucleus sampling — diversidade lexical
    #top_p=0.9,

    # Top-K sampling — estabilidade
    #top_k=40,

    # Evita repetição excessiva
    #repeat_penalty=1.1,
)


def analyze_image_stream(image_path: str, question: str):
    image_b64 = image_to_base64(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                },
            },
        ]
    )

    for chunk in llm_vision.stream([message]):
        if chunk.content:
            yield chunk.content

# ==================================================
# Chain RAG (STREAM)
# ==================================================
chain = (
    {
        "context": retrieve_context,
        "input": lambda x: x["input"],
        "history": lambda x: history_formatter.invoke(x["history"]),
    }
    | prompt
    | llm_text
    | StrOutputParser()
)

runnable = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def chat_with_llm(session_id: str, user_input: str):
    for chunk in runnable.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk

# ==================================================
# Chat
# ==================================================
user_prompt = st.chat_input("Pergunte algo sobre documentos ou desenhos...")

if user_prompt:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # -------- IMAGEM → VISION STREAM --------
    if st.session_state.image_docs:
        with st.chat_message("assistant"):
            response = st.write_stream(
                analyze_image_stream(
                    st.session_state.image_docs[-1]["path"],
                    user_prompt,
                )
            )

    # -------- TEXTO → RAG STREAM --------
    else:
        with st.chat_message("assistant"):
            response = st.write_stream(
                chat_with_llm(user_id, user_prompt)
            )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )
