"""
Módulo responsável pela análise de imagens (multimodal).
"""

import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from config.settings import BASE_URL, MODEL_VISION

def image_to_base64(path: str) -> str:
    """
    Converte imagem local para Base64 (exigido pelo Ollama).
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_vision_llm():
    """
    Inicializa o modelo multimodal.
    """
    return ChatOllama(
        base_url=BASE_URL,
        model=MODEL_VISION,
        temperature=0.0,  # descrição objetiva
    )

def analyze_image_stream(image_path: str, question: str):
    """
    Analisa imagem + pergunta em modo streaming.
    """
    llm = get_vision_llm()
    image_b64 = image_to_base64(image_path)

    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            },
        },
    ])

    for chunk in llm.stream([message]):
        if chunk.content:
            yield chunk.content
