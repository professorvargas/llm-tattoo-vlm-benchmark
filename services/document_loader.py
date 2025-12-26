"""
Responsável exclusivamente pelo carregamento de documentos.

Princípio da Responsabilidade Única (SRP):
- Lê arquivos
- Converte para objetos LangChain
"""

import os
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)

def load_documents(files):
    """
    Processa arquivos enviados pelo usuário.

    Retorna:
    - text_docs: documentos textuais (PDF/DOCX)
    - image_docs: imagens para análise multimodal
    """
    text_docs = []
    image_docs = []

    for file in files:
        suffix = file.name.split(".")[-1].lower()

        # Salva temporariamente o arquivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # --- Tipos suportados ---
        if suffix == "pdf":
            text_docs.extend(PyPDFLoader(tmp_path).load())

        elif suffix == "docx":
            text_docs.extend(Docx2txtLoader(tmp_path).load())

        elif suffix in ["jpg", "jpeg", "png"]:
            image_docs.append({
                "path": tmp_path,
                "name": file.name
            })
        else:
            os.remove(tmp_path)

    return text_docs, image_docs
