import base64
import os
from openai import OpenAI


def verify_with_vision(
    image_bytes: bytes,
    question: str,
    answer: str,
):
    """
    Uses GPT-4o Vision to verify whether the highlighted image
    actually supports the given answer.
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Convert image → base64 → data URL
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/png;base64,{image_b64}"

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are an audit verification assistant.\n\n"
                            f"Question: {question}\n"
                            f"Answer given: {answer}\n\n"
                            "Check whether the highlighted document image "
                            "visually supports the answer.\n\n"
                            "Reply strictly in one of the following formats:\n"
                            "- SUPPORTED: <short reason>\n"
                            "- NOT SUPPORTED: <short reason>"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,  # ✅ STRING ONLY
                    },
                ],
            }
        ],
        max_output_tokens=150,
    )

    return response.output_text
