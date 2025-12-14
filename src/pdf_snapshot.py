import fitz
from PIL import Image, ImageDraw
import io


def render_highlight(pdf_path, page_number, box, zoom=2.0):
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes()))
    draw = ImageDraw.Draw(img, "RGBA")

    draw.rectangle(
        [
            box["x0"] * zoom,
            box["y0"] * zoom,
            box["x1"] * zoom,
            box["y1"] * zoom,
        ],
        fill=(255, 255, 0, 80),
        outline=(255, 180, 0, 200),
        width=3,
    )

    return img
