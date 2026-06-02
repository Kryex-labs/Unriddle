import pytesseract
from pdf2image import convert_from_path
import json
import re
import os
from pathlib import Path
from project_paths import PROJECT_ROOT, PATIENT_PDF, PATIENT_TEXT

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
POPPLER_PATH = os.getenv("POPPLER_PATH")

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

PDF_PATH = str(PATIENT_PDF)
OUTPUT_DIR = PROJECT_ROOT

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def get_total_pages():
    from pypdf import PdfReader
    reader = PdfReader(PDF_PATH)
    return len(reader.pages)

def extract_with_ocr():
    total = get_total_pages()
    print(f"Total pages: {total}")

    # Resume support: load existing progress if any
    progress_path = OUTPUT_DIR / "patient_data_progress.json"
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            all_pages = json.load(f)
        done = {p["page"] for p in all_pages}
        print(f"Resuming — {len(done)} pages already done")
    else:
        all_pages = []
        done = set()

    for page_num in range(1, total + 1):
        if page_num in done:
            continue

        print(f"  OCR page {page_num}/{total}...", flush=True)
        try:
            # Convert single page at a time — low memory
            images = convert_from_path(
                PDF_PATH,
                dpi=200,
                first_page=page_num,
                last_page=page_num,
                poppler_path=POPPLER_PATH or None
            )
            raw_text = pytesseract.image_to_string(images[0], lang='eng')
            text = clean_text(raw_text)
            del images  # free memory immediately
        except Exception as e:
            print(f"  ERROR on page {page_num}: {e}")
            text = f"[OCR FAILED: {e}]"

        all_pages.append({"page": page_num, "text": text})
        all_pages.sort(key=lambda x: x["page"])

        # Save progress after every page
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(all_pages, f, indent=2, ensure_ascii=False)

    print("\nAll pages done. Writing final output files...")

    # Write TXT
    txt_path = PATIENT_TEXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"PATIENT DATA EXTRACTION (OCR)\nSource: {PDF_PATH}\nTotal Pages: {total}\n\n")
        for p in all_pages:
            f.write(f"{'='*60}\nPAGE {p['page']}\n{'='*60}\n{p['text']}\n\n")
    print(f"TXT saved: {txt_path}  ({txt_path.stat().st_size/1024:.1f} KB)")

    # Write JSON
    json_path = PROJECT_ROOT / "patient_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"source": "patient_data.pdf", "total_pages": total, "pages": all_pages}, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}  ({json_path.stat().st_size/1024:.1f} KB)")

    # Stats
    total_chars = sum(len(p["text"]) for p in all_pages)
    pages_with_text = sum(1 for p in all_pages if len(p["text"]) > 50)
    print(f"\nStats: {pages_with_text}/{total} pages with text, {total_chars:,} total chars")

    # Preview first page with content
    for p in all_pages:
        if len(p["text"]) > 50:
            print(f"\n--- Page {p['page']} preview ---")
            print(p["text"][:500])
            break

if __name__ == "__main__":
    extract_with_ocr()
