import base64
from pathlib import Path

# ==================================================
# LangChain / Ollama
# ==================================================
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ==================================================
# Configuration
# ==================================================
BASE_URL = "http://localhost:11434"

MODEL_VISION = "gemma3:12b"

# ==================================================
# Utilities
# ==================================================
def image_to_base64(path: str) -> str:
    """
    Converts a local image to base64 (format required by Ollama).
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==================================================
# Image path (VARIABLE)
# ==================================================
IMAGE_PATH = "./datasets/tattoo1.png"

# Simple validation
if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_b64 = image_to_base64(IMAGE_PATH)

# ==================================================
# LLM Vision (Ollama)
# ==================================================
llm_vision = ChatOllama(
    model=MODEL_VISION,
    base_url=BASE_URL,
    temperature=0.0,
)

# ==================================================
# Multimodal message
# ==================================================
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Analyse the image and describe in detail what you see."
        },
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_b64}"
        }
    ]
)

# ==================================================
# Model call
# ==================================================
for chunk in llm_vision.stream([message]):
    if chunk.content:
        print(chunk.content, end="", flush=True)
