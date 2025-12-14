import os
from langchain_openai import OpenAIEmbeddings

_embeddings_instance = None


def get_embeddings():
    global _embeddings_instance

    if _embeddings_instance:
        return _embeddings_instance

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Set it via environment variables or Streamlit secrets."
        )

    # 🔑 Force env var so LangChain/OpenAI always sees it
    os.environ["OPENAI_API_KEY"] = api_key

    _embeddings_instance = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return _embeddings_instance
