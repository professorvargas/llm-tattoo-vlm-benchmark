"""
Componentes visuais da sidebar.
"""

import streamlit as st

def render_sidebar():
    """
    Renderiza a área de upload de documentos.
    """
    st.sidebar.header("📂 Base de Conhecimento")

    return st.sidebar.file_uploader(
        "Envie PDF, DOCX ou IMAGENS",
        type=["pdf", "docx", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
