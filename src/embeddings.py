import os
from langchain_openai import OpenAIEmbeddings


def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing at runtime")

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,   # 👈 EXPLICIT
    )
