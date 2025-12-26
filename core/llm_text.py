"""
Configuração do LLM textual utilizado no RAG.

Todos os hiperparâmetros ficam centralizados aqui.
"""

from langchain_ollama import ChatOllama
from config.settings import BASE_URL, MODEL_TEXT

def get_text_llm():
    """
    Retorna um LLM configurado para respostas determinísticas.
    """
    return ChatOllama(
        base_url=BASE_URL,
        model=MODEL_TEXT,

        # Criatividade controlada
        temperature=0.0,
        top_p=0.9,
        top_k=40,

        # Controle de repetição
        repeat_penalty=1.15,

        # Limites de contexto
        num_ctx=4096,
        num_predict=512,
    )
