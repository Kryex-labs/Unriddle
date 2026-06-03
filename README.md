# Dscribe AI — Backend

> Agentic AI system for clinical discharge summaries with live Pinecone learning loop.  
> Built with **FastAPI · Claude Opus 4.5 · Pinecone · Jina AI**  
> Deployed on **Railway** · Frontend on **Vercel**

---

## What This Is

The backend for the **Dscribe AI Discharge Summary Agent** — a take-home assignment for the AI Engineer role at Unriddle Technologies.

**What it does:**
1. Reads a 71-page scanned PDF of synthetic patient records via OCR (Tesseract)
2. Runs a **Claude Opus 4.5 agentic loop** with 7 clinical tools to extract a structured discharge summary
3. Enforces **zero fabrication** — every field is either sourced from documents or explicitly marked `[MISSING]`
4. Detects conflicts between documents, reconciles medications, flags escalations
5. After every summary is generated, **automatically runs a background learning loop**: a simulated doctor reviews the draft, corrections are extracted and stored permanently in Pinecone — improving every future run

---

## Live URLs

| Service | URL |
|---------|-----|
| Backend API | https://unriddle-production.up.railway.app |
| API Docs | https://unriddle-production.up.railway.app/docs |
| Frontend | Deployed on Vercel |

---

## Architecture

```
PDF (71 pages)
    ↓ OCR (Tesseract + pdf2image, page-by-page)
patient_data.json
    ↓
Claude Opus 4.5 Agent (Azure endpoint)
    ↓ uses 7 tools:
    list_patient_pages · read_pages · reconcile_medications
    detect_conflict · check_drug_interaction
    escalate_to_clinician · flag_missing_field
    ↓
Structured Discharge Summary (JSON + text)
    ↓ background thread
Simulated Doctor Review (Claude Opus 4.5)
    ↓
Correction Patterns → Jina AI Embeddings (1024-dim) → Pinecone Cloud
    ↓ injected into next run
Better future summaries
```

---

## Project Structure

```
discharge_agent/
├── agent.py            # Claude Opus 4.5 agent loop (MAX_STEPS=15)
├── claude_client.py    # Azure Claude API client with retry logic
├── models.py           # Pydantic models: DischargeSummary, ClinicalField, Medication
└── tools.py            # All 7 agent tools

learning/
├── correction_store.py # Pinecone + Jina AI: store/query correction vectors
├── reviewer.py         # Simulated doctor (Claude with hidden editing policy)
├── reward.py           # Section-level edit distance metric
└── run_loop.py         # Batch learning orchestrator (3 iterations)

api/
└── main.py             # FastAPI server: all endpoints + background learning

ocr_extract.py          # Page-by-page Tesseract OCR pipeline
patient_data.json       # Extracted OCR text (71 pages, 22K chars)
project_paths.py        # Cross-platform path resolution
requirements.txt        # Python dependencies
railway.toml            # Railway deployment config
```

---

## API Endpoints

### Core Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + live Pinecone count |
| `GET` | `/patients` | List available patients |
| `POST` | `/run` | Start agent for a patient (async, returns `job_id`) |
| `GET` | `/run/{job_id}` | Poll job status (`running` / `done` / `error`) |
| `GET` | `/summary/{patient_id}` | Fetch completed discharge summary (JSON + text) |
| `GET` | `/trace/{patient_id}` | Fetch agent step trace |

### Learning Loop

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/learning/status` | **Live learning state** — poll this to show real-time activity |
| `GET` | `/learning/events` | Recent learning events (one per run, newest first) |
| `GET` | `/learning/corrections` | All correction vectors stored in Pinecone |
| `GET` | `/learning/metrics` | Batch learning metrics (reward per iteration) |
| `GET` | `/learning/improvement` | Before/after improvement summary |
| `POST` | `/learning/run` | Trigger full batch learning loop (offline use) |

---

## How the Learning Loop Works

Every time a summary is generated via `POST /run`:

1. **Summary saved → job marked `done`** — user gets their result immediately
2. **Background thread starts** (user doesn't wait):
   - Simulated doctor reviews the draft using Claude Opus 4.5 with a hidden editing policy
   - Corrections extracted: what changed + why (generalizable patterns)
   - Each correction embedded with Jina AI (1024-dim) and stored in Pinecone
3. **Next run is better**: top-5 relevant corrections from Pinecone are injected into the agent's system prompt

**Learning is cumulative and permanent.** Pinecone persists corrections across all restarts and deployments. Cold-start is solved.

### Doctor's Hidden Editing Policy
The simulated reviewer consistently applies rules the agent doesn't know (must learn from corrections):
- Expand dosing abbreviations: `1-0-0` → `once daily (morning)`
- Always include route of administration: `oral / IV / subcutaneous`
- Format diagnoses: `Condition — clinical qualifier`
- Standard follow-up: `Return to ED immediately if symptoms worsen`
- Pending results note: `Results to be communicated by treating team`

---

## Running Locally

### Prerequisites
- Python 3.10+
- Tesseract OCR (`winget install UB-Mannheim.TesseractOCR`)
- Poppler (`winget install oschwartz10612.poppler`)

### Setup

```bash
git clone <this-repo>
cd unriddle_technologies

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Fill in your API keys (see Environment Variables below)

# Run OCR extraction (first time only)
python ocr_extract.py

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs

### Run the agent directly (CLI)

```bash
python main.py
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CLAUDE_API_KEY` | Yes | Azure Claude Opus 4.5 API key |
| `CLAUDE_ENDPOINT` | Yes | Azure Claude endpoint URL |
| `CLAUDE_MODEL` | Yes | `claude-opus-4-5` |
| `PINECONE_API_KEY` | Yes | Pinecone serverless API key |
| `PINECONE_INDEX` | Yes | `discharge-corrections` |
| `JINA_API_KEY` | Yes | Jina AI embedding API key |

Create a `.env` file with these values. **Never commit `.env` to git.**

---

## Part 1 — Agent Design

The agent uses Claude Opus 4.5 with a hard step cap (`MAX_STEPS = 15`). Every step logs:
```
reasoning → tool chosen → inputs → result → next decision
```

**The 7 tools:**

| Tool | Purpose |
|------|---------|
| `list_patient_pages` | Survey all docs with type classification |
| `read_pages` | Read full text of specific pages (max 5/call) |
| `reconcile_medications` | Formal diff: admission vs discharge meds |
| `detect_conflict` | Record contradictions between documents |
| `check_drug_interaction` | Mock interaction DB (9 known pairs) |
| `escalate_to_clinician` | Flag safety concerns to review queue |
| `flag_missing_field` | Mark required fields as MISSING/PENDING |

**No fabrication guarantee:** Every output field is either sourced from a document (with page citation) or explicitly marked `[MISSING]`, `[PENDING]`, or `[CONFLICT]`. The system prompt uses `"NEVER invent, guess, or infer"` as an absolute rule enforced in Pydantic models.

---

## Part 2 — Learning Loop Results

| Metric | Value |
|--------|-------|
| Baseline reward (iter 0) | 75.8% |
| Peak reward (Patient 1, iter 1) | 82.9% (+26.2%) |
| Corrections in Pinecone | 61 (and growing with every run) |
| Embedding model | Jina AI `jina-embeddings-v3` (1024-dim) |
| Vector DB | Pinecone Serverless (AWS us-east-1, free tier) |

**Limitations observed:**
- Cross-patient bleed: corrections from gastroenteritis patient degraded DKA patient
- Small dataset (2 patients) — production needs 100+ pairs
- Gaming risk: edit distance rewards can be gamed by vagueness (guarded by completeness scoring)

---

## Deployment (Railway)

Railway auto-deploys from GitHub on every push.

```toml
# railway.toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
```

**Required Railway environment variables:** Set all variables from the table above in Railway → Project → Variables.
