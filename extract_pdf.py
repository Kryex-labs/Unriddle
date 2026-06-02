"""
Extract all content from patient_data.pdf into a clean structured text file.
Handles text-based and image-heavy pages using PyMuPDF.
"""

import fitz  # PyMuPDF
import pdfplumber
import json
import re
from pathlib import Path
from project_paths import PATIENT_PDF, PATIENT_TEXT_CLEAN, PATIENT_PAGES_JSON

PDF_PATH = PATIENT_PDF
OUT_TXT = PATIENT_TEXT_CLEAN
OUT_JSON = PATIENT_PAGES_JSON


def clean_text(text: str) -> str:
    if not text:
        return ""
    # collapse excessive whitespace / blank lines
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_with_pdfplumber(path: Path):
    """Primary extraction: pdfplumber preserves layout and tables well."""
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages, 1):
            print(f"  Extracting page {i}/{len(pdf.pages)}...", end="\r")
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

            # Also grab tables
            tables_raw = page.extract_tables() or []
            table_strs = []
            for tbl in tables_raw:
                rows = []
                for row in tbl:
                    cleaned_row = [cell.strip() if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned_row))
                table_strs.append("\n".join(rows))

            pages.append({
                "page": i,
                "text": clean_text(text),
                "tables": table_strs,
            })
    print()
    return pages


def extract_with_pymupdf(path: Path):
    """Fallback extraction using PyMuPDF — good for messy layouts."""
    pages = []
    doc = fitz.open(str(path))
    print(f"Total pages (PyMuPDF): {len(doc)}")
    for i, page in enumerate(doc, 1):
        print(f"  PyMuPDF page {i}/{len(doc)}...", end="\r")
        blocks = page.get_text("blocks")
        # blocks: (x0, y0, x1, y1, text, block_no, block_type)
        # sort top-to-bottom, left-to-right
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 10), b[0]))
        text = "\n".join(b[4].strip() for b in blocks_sorted if b[6] == 0 and b[4].strip())
        pages.append({"page": i, "text": clean_text(text), "tables": []})
    print()
    doc.close()
    return pages


def merge_extractions(plumber_pages, mupdf_pages):
    """Use pdfplumber text when substantial, fall back to pymupdf."""
    merged = []
    for pb, mu in zip(plumber_pages, mupdf_pages):
        text = pb["text"] if len(pb["text"]) > len(mu["text"]) else mu["text"]
        merged.append({
            "page": pb["page"],
            "text": text,
            "tables": pb["tables"],
        })
    return merged


def build_clean_document(pages):
    """Assemble a single clean text document from all pages."""
    sections = []
    for p in pages:
        header = f"\n{'='*60}\n PAGE {p['page']}\n{'='*60}\n"
        body = p["text"]
        if p["tables"]:
            body += "\n\n[TABLES]\n" + "\n\n---\n".join(p["tables"])
        sections.append(header + body)
    return "\n".join(sections)


def detect_patients(pages):
    """Best-effort: detect patient boundaries by looking for patient headers."""
    patient_markers = []
    for p in pages:
        text_upper = p["text"].upper()
        if any(kw in text_upper for kw in ["PATIENT NAME", "PATIENT ID", "MRN", "ADMISSION DATE", "DOB:"]):
            patient_markers.append(p["page"])
    return patient_markers


if __name__ == "__main__":
    print("=== PDF Extraction Started ===")

    print("\n[1/4] Extracting with pdfplumber...")
    plumber_pages = extract_with_pdfplumber(PDF_PATH)

    print("[2/4] Extracting with PyMuPDF...")
    mupdf_pages = extract_with_pymupdf(PDF_PATH)

    print("[3/4] Merging extractions...")
    pages = merge_extractions(plumber_pages, mupdf_pages)

    print("[4/4] Writing output files...")
    clean_doc = build_clean_document(pages)
    OUT_TXT.write_text(clean_doc, encoding="utf-8")
    print(f"  -> Clean text: {OUT_TXT}")

    OUT_JSON.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> JSON pages: {OUT_JSON}")

    # Quick stats
    total_chars = sum(len(p["text"]) for p in pages)
    empty_pages = [p["page"] for p in pages if len(p["text"]) < 50]
    patient_pages = detect_patients(pages)

    print("\n=== Extraction Summary ===")
    print(f"  Total pages     : {len(pages)}")
    print(f"  Total chars     : {total_chars:,}")
    print(f"  Sparse pages    : {empty_pages}")
    print(f"  Patient markers : pages {patient_pages}")
    print("\nDone.")
