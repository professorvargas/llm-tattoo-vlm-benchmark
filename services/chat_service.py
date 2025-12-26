"""
Serviço responsável pelo streaming do chat textual.
"""

def chat_stream(runnable, session_id: str, user_input: str):
    """
    Executa a cadeia RAG em modo streaming.
    """
    for chunk in runnable.stream(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk
