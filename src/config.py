import os


def get_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it via environment variables or Streamlit secrets."
        )

    os.environ["OPENAI_API_KEY"] = key
    return key
