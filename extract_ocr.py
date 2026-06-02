"""
OCR extraction using PyMuPDF to render pages as images,
then Claude Vision API to extract text from each page.
"""

import fitz  # PyMuPDF
import anthropic
import base64
import json
import os
import time
from pathlib import Path
from project_paths import PROJECT_ROOT, PATIENT_PDF, PATIENT_TEXT_CLEAN, PATIENT_PAGES_JSON

PDF_PATH = PATIENT_PDF
OUT_DIR = PROJECT_ROOT / "pages"
OUT_TXT = PATIENT_TEXT_CLEAN
OUT_JSON = PATIENT_PAGES_JSON

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

OUT_DIR.mkdir(exist_ok=True)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def render_page_as_image(doc, page_num: int, dpi: int = 200) -> bytes:
    """Render a PDF page to PNG bytes."""
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def ocr_page_with_claude(image_bytes: bytes, page_num: int) -> str:
    """Send page image to Claude Vision and extract all text."""
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = """You are a medical document OCR system. Extract ALL text from this clinical document image exactly as it appears.

Rules:
- Preserve section headers, labels, and structure
- Keep table rows intact with | separators
- Mark unclear/unreadable text as [ILLEGIBLE]
- Do NOT interpret, summarize, or omit anything
- Include every field name, value, date, and number you can see
- Preserve newlines between sections

Output ONLY the extracted text, nothing else."""

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    return response.content[0].text


def main():
    doc = fitz.open(str(PDF_PATH))
    total_pages = len(doc)
    print(f"PDF has {total_pages} pages (all scanned images — using Claude Vision OCR)\n")

    pages_data = []
    # Check if we have partial results already
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        done_pages = {p["page"] for p in existing}
        pages_data = existing
        print(f"Resuming: {len(done_pages)} pages already done\n")
    else:
        done_pages = set()

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        if page_num in done_pages:
            print(f"  [SKIP] Page {page_num} already extracted")
            continue

        print(f"  [OCR ] Page {page_num}/{total_pages}...", end=" ", flush=True)
        try:
            img_bytes = render_page_as_image(doc, page_idx, dpi=200)
            # Save image for reference
            img_path = OUT_DIR / f"page_{page_num:03d}.png"
            img_path.write_bytes(img_bytes)

            text = ocr_page_with_claude(img_bytes, page_num)
            pages_data.append({"page": page_num, "text": text, "tables": []})
            print(f"OK ({len(text)} chars)")

            # Save progress after every page
            pages_data_sorted = sorted(pages_data, key=lambda x: x["page"])
            OUT_JSON.write_text(json.dumps(pages_data_sorted, indent=2, ensure_ascii=False), encoding="utf-8")

            # Small delay to avoid rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"ERROR: {e}")
            pages_data.append({"page": page_num, "text": f"[EXTRACTION_ERROR: {e}]", "tables": []})

    doc.close()

    # Build final clean text doc
    pages_data_sorted = sorted(pages_data, key=lambda x: x["page"])
    lines = []
    for p in pages_data_sorted:
        lines.append(f"\n{'='*60}")
        lines.append(f" PAGE {p['page']}")
        lines.append(f"{'='*60}\n")
        lines.append(p["text"])

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    total_chars = sum(len(p["text"]) for p in pages_data_sorted)
    print(f"\n=== Done ===")
    print(f"  Pages extracted : {len(pages_data_sorted)}")
    print(f"  Total chars     : {total_chars:,}")
    print(f"  Clean text      : {OUT_TXT}")
    print(f"  JSON            : {OUT_JSON}")


if __name__ == "__main__":
    main()
