from langchain_openai import OpenAIEmbeddings
from .config import get_openai_key

def get_embeddings():
    return OpenAIEmbeddings(
        api_key=get_openai_key()
    )
