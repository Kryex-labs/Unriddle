"""
Shared filesystem paths for the discharge summary project.

Keep all deploy-sensitive paths relative to the repository root so the
code runs unchanged on Windows, Railway, or any other host.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("UNRIDDLE_ROOT", Path(__file__).resolve().parent)).resolve()
OUTPUT_DIR = Path(os.getenv("UNRIDDLE_OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
ITERATIONS_DIR = Path(os.getenv("UNRIDDLE_ITERATIONS_DIR", OUTPUT_DIR / "iterations")).resolve()
PATIENT_DATA_JSON = Path(os.getenv("PATIENT_DATA_JSON_PATH", PROJECT_ROOT / "patient_data.json")).resolve()
PATIENT_PDF = Path(os.getenv("PATIENT_PDF_PATH", PROJECT_ROOT / "patient_data.pdf")).resolve()
PATIENT_TEXT = Path(os.getenv("PATIENT_TEXT_PATH", PROJECT_ROOT / "patient_data.txt")).resolve()
PATIENT_PAGES_JSON = Path(os.getenv("PATIENT_PAGES_JSON_PATH", PROJECT_ROOT / "patient_data_pages.json")).resolve()
PATIENT_TEXT_CLEAN = Path(os.getenv("PATIENT_TEXT_CLEAN_PATH", PROJECT_ROOT / "patient_data_clean.txt")).resolve()

