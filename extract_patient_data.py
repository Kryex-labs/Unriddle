import pdfplumber
import json
import re
from pathlib import Path
from project_paths import PROJECT_ROOT, PATIENT_PDF, PATIENT_TEXT

PDF_PATH = str(PATIENT_PDF)
OUTPUT_DIR = PROJECT_ROOT

def clean_text(text):
    if not text:
        return ""
    # normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_pdf():
    all_pages = []
    full_text_parts = []

    print(f"Opening PDF: {PDF_PATH}")
    with pdfplumber.open(PDF_PATH) as pdf:
        total = len(pdf.pages)
        print(f"Total pages: {total}")

        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            print(f"  Processing page {page_num}/{total}...", end="\r")

            # Extract text
            raw_text = page.extract_text() or ""
            text = clean_text(raw_text)

            # Extract tables
            tables = []
            raw_tables = page.extract_tables()
            for tbl in (raw_tables or []):
                cleaned = []
                for row in tbl:
                    cleaned.append([cell if cell else "" for cell in row])
                tables.append(cleaned)

            page_data = {
                "page": page_num,
                "text": text,
                "tables": tables
            }
            all_pages.append(page_data)

            # Build full text block
            page_block = f"{'='*60}\nPAGE {page_num}\n{'='*60}\n{text}"
            if tables:
                page_block += "\n\n[TABLES ON THIS PAGE]\n"
                for t_idx, tbl in enumerate(tables):
                    page_block += f"\nTable {t_idx+1}:\n"
                    for row in tbl:
                        page_block += "  | " + " | ".join(str(c) for c in row) + " |\n"
            full_text_parts.append(page_block)

    print(f"\nExtraction complete. Writing output files...")

    # --- Write TXT ---
    txt_path = PATIENT_TEXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"PATIENT DATA EXTRACTION\nSource: {PDF_PATH}\nTotal Pages: {total}\n\n")
        f.write("\n\n".join(full_text_parts))
    print(f"TXT saved: {txt_path}")

    # --- Write JSON ---
    json_path = PROJECT_ROOT / "patient_data.json"
    output = {
        "source": "patient_data.pdf",
        "total_pages": total,
        "pages": all_pages
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {json_path}")

    # Stats
    total_chars = sum(len(p["text"]) for p in all_pages)
    total_tables = sum(len(p["tables"]) for p in all_pages)
    pages_with_text = sum(1 for p in all_pages if p["text"])
    print(f"\nStats:")
    print(f"  Pages with text: {pages_with_text}/{total}")
    print(f"  Total tables found: {total_tables}")
    print(f"  Total characters: {total_chars:,}")

if __name__ == "__main__":
    extract_pdf()
