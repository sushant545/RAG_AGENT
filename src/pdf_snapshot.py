import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io


def generate_highlight_snapshot(
    pdf_path: str,
    page_number: int,
    highlight_text: str,
    zoom: float = 2.0
):
    """
    Returns a PIL Image with highlighted text on the given PDF page.
    """

    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # Search text occurrences
    text_instances = page.search_for(highlight_text)

    if not text_instances:
        # fallback: return page without highlight
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return Image.open(io.BytesIO(pix.tobytes()))

    # Render page
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes()))
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw highlight rectangles
    for rect in text_instances:
        scaled_rect = fitz.Rect(
            rect.x0 * zoom,
            rect.y0 * zoom,
            rect.x1 * zoom,
            rect.y1 * zoom,
        )

        draw.rectangle(
            [scaled_rect.x0, scaled_rect.y0, scaled_rect.x1, scaled_rect.y1],
            fill=(255, 255, 0, 80),  # yellow transparent
            outline=(255, 200, 0, 200),
            width=2,
        )

    return img
