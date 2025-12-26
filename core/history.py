"""
Gerenciamento do histórico de conversas.

Utiliza SQLite + SQLChatMessageHistory para permitir:
- Persistência entre sessões
- RAG com memória de contexto
"""

from sqlalchemy import create_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from config.settings import DB_URL

# Engine SQLAlchemy reutilizável
engine = create_engine(DB_URL)

def get_session_history(session_id: str):
    """
    Retorna o histórico associado a um usuário/sessão.
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=engine,
    )
