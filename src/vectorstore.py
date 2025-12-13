# src/vectorstore.py

from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings


def create_vectorstore(documents):
    if not documents:
        raise ValueError("No documents provided to create vectorstore")

    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)
