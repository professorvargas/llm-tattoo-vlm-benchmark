"""
Construção da cadeia RAG (Retriever + Prompt + LLM).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.llm_text import get_text_llm
from core.history import get_session_history

def build_rag_chain(retriever):
    """
    Monta a cadeia RAG com memória persistente.
    """

    retrieve_context = RunnableLambda(
        lambda x: retriever.invoke(x["input"])
    )

    prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado.

Use APENAS o CONTEXTO para responder.
Se não souber, diga claramente que não encontrou.

CONTEXTO:
{context}

HISTÓRICO:
{history}

PERGUNTA:
{input}

RESPOSTA:
""")

    chain = (
        {
            "context": retrieve_context,
            "input": lambda x: x["input"],
            "history": lambda x: x["history"],
        }
        | prompt
        | get_text_llm()
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
