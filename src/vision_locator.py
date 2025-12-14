import base64
import fitz  # PyMuPDF
import io
import json
import os
import re
from PIL import Image
from openai import OpenAI

# -------------------------------------------------------------------
# OpenAI client
# -------------------------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def locate_evidence(
    pdf_path: str,
    page_number: int,
    question: str,
    answer: str,
):
    """
    Uses GPT-4o Vision to locate the exact visual evidence
    (bounding box) on a PDF page that supports the answer.

    Returns:
        {
          "x0": float,
          "y0": float,
          "x1": float,
          "y1": float,
          "confidence": float
        }
    """

    # ----------------------------------------------------------------
    # Render PDF page to image
    # ----------------------------------------------------------------
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    zoom = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    image = Image.open(io.BytesIO(pix.tobytes()))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()

    # ----------------------------------------------------------------
    # Vision prompt (STRICT, audit-grade)
    # ----------------------------------------------------------------
    prompt = f"""
You are an audit assistant.

The user asked:
{question}

The system answered:
{answer}

Look at the document image and identify the EXACT visual region
that proves the answer.

Return ONLY a JSON object in this exact format:
{{
  "x0": number,
  "y0": number,
  "x1": number,
  "y1": number,
  "confidence": number
}}

Rules:
- Do NOT explain
- Do NOT add text
- Do NOT wrap in markdown
"""

    # ----------------------------------------------------------------
    # Call GPT-4o Vision
    # ----------------------------------------------------------------
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded_image}",
                    },
                ],
            }
        ],
    )

    # ----------------------------------------------------------------
    # Extract text output safely (SDK OBJECT MODEL)
    # ----------------------------------------------------------------
    output_text = ""

    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    output_text += content.text

    output_text = output_text.strip()

    if not output_text:
        raise RuntimeError("Vision model returned empty output")

    # ----------------------------------------------------------------
    # Extract JSON from free text (robust)
    # ----------------------------------------------------------------
    match = re.search(r"\{.*\}", output_text, re.DOTALL)

    if not match:
        raise RuntimeError(
            f"Vision output did not contain JSON:\n{output_text}"
        )

    json_text = match.group(0)

    try:
        result = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse Vision JSON:\n{json_text}"
        ) from e

    # ----------------------------------------------------------------
    # Validate expected fields (audit safety)
    # ----------------------------------------------------------------
    required_keys = {"x0", "y0", "x1", "y1", "confidence"}
    if not required_keys.issubset(result):
        raise RuntimeError(
            f"Vision JSON missing required keys:\n{result}"
        )

    return result
