"""
FastAPI server — Discharge Summary Agent + Live Learning Loop
Deployed on Railway. Frontend (Vercel) calls these endpoints via proxy.
"""
import sys
import json
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from discharge_agent.agent import DischargeAgent
from learning.correction_store import get_correction_count, get_all_corrections
from learning.run_loop import build_learned_context
from project_paths import OUTPUT_DIR

app = FastAPI(
    title="Discharge Summary Agent",
    description="Agentic AI for clinical discharge summaries with live Pinecone learning loop",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_FILE = OUTPUT_DIR / "learning_events.json"

_jobs: dict = {}

# Global learning state — tracks what's happening right now
_learning_state = {
    "running": False,
    "patient_id": None,
    "stage": None,           # "doctor_review" | "extracting" | "storing"
    "started_at": None,
}


# ── Helpers ──────────────────────────────────────────────────

def _load_events() -> list:
    if EVENTS_FILE.exists():
        try:
            return json.loads(EVENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_event(event: dict):
    events = _load_events()
    events.insert(0, event)          # newest first
    events = events[:50]             # keep last 50
    EVENTS_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")


def _run_background_learning(patient_id: str, draft_text: str):
    """
    Runs AFTER the summary is already saved and job is marked 'done'.
    User sees their summary immediately. This runs silently in background.
    Never raises — any failure is logged but ignored.
    """
    global _learning_state
    _learning_state.update({"running": True, "patient_id": patient_id, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    event = {
        "patient_id": patient_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "started",
        "corrections_extracted": 0,
        "total_in_db": get_correction_count(),
        "reward": None,
    }

    try:
        from learning.reviewer import review_draft, extract_corrections
        from learning.reward import compute_reward

        # Stage 1 — Simulated doctor reviews the draft
        _learning_state["stage"] = "doctor_review"
        edited_text = review_draft(draft_text, patient_id)

        # Stage 2 — Compute reward (how much the doctor had to change)
        reward_data = compute_reward(draft_text, edited_text)
        event["reward"] = round(reward_data["overall_reward"], 4)
        event["edit_burden"] = round(reward_data["overall_edit_burden"], 4)
        event["interpretation"] = reward_data["interpretation"]

        # Stage 3 — Extract generalizable correction patterns
        _learning_state["stage"] = "extracting"
        corrections = extract_corrections(draft_text, edited_text, patient_id, iteration=0)

        # Stage 4 — Store in Pinecone
        _learning_state["stage"] = "storing"
        if corrections:
            from learning.correction_store import store_corrections_batch
            store_corrections_batch(corrections)

        event["corrections_extracted"] = len(corrections)
        event["total_in_db"] = get_correction_count()
        event["status"] = "completed"

        # Save edited draft for reference
        edited_path = OUTPUT_DIR / f"{patient_id.lower()}_last_edited.txt"
        edited_path.write_text(edited_text, encoding="utf-8")

        print(f"[Learning] {patient_id}: +{len(corrections)} corrections → {event['total_in_db']} total in Pinecone | reward={event['reward']}")

    except Exception as e:
        event["status"] = "error"
        event["error"] = str(e)
        print(f"[Learning] Background learning failed for {patient_id}: {e}")
    finally:
        _save_event(event)
        _learning_state.update({"running": False, "patient_id": None, "stage": None, "started_at": None})


# ── Request models ────────────────────────────────────────────

class RunRequest(BaseModel):
    patient_id: str
    use_learned_corrections: bool = True


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "claude-opus-4-5",
        "corrections_in_pinecone": get_correction_count(),
        "learning_active": _learning_state["running"],
        "message": "Discharge Summary Agent is live"
    }


@app.get("/patients")
def list_patients():
    return {"patients": [
        {"id": "PATIENT_1", "description": "Acute Gastroenteritis + UTI + Thyroid disorder"},
        {"id": "PATIENT_2", "description": "DKA + Uncontrolled T2DM + Acute Pyelonephritis"}
    ]}


@app.post("/run")
def run_agent(req: RunRequest, background_tasks: BackgroundTasks):
    if req.patient_id not in ["PATIENT_1", "PATIENT_2"]:
        raise HTTPException(400, "patient_id must be PATIENT_1 or PATIENT_2")

    job_id = f"{req.patient_id}_{int(time.time())}"
    _jobs[job_id] = {"status": "running", "patient_id": req.patient_id}

    def _run():
        try:
            # ── Phase 1: Generate discharge summary ──────────────
            learned = build_learned_context() if req.use_learned_corrections else ""
            agent   = DischargeAgent(extra_system_context=learned)
            trace   = str(OUTPUT_DIR / f"{req.patient_id.lower()}_trace.txt")
            summary = agent.run(patient_id=req.patient_id, trace_path=trace)

            draft_text = summary.to_readable()
            (OUTPUT_DIR / f"{req.patient_id.lower()}_summary.txt").write_text(draft_text, encoding="utf-8")
            (OUTPUT_DIR / f"{req.patient_id.lower()}_summary.json").write_text(
                json.dumps(summary.model_dump(), indent=2, default=str), encoding="utf-8")

            # Mark job done — user gets their summary NOW
            _jobs[job_id] = {"status": "done", "patient_id": req.patient_id}

            # ── Phase 2: Background learning (non-blocking) ───────
            # Runs in a separate daemon thread so it never blocks the job.
            # If this fails, the summary is already saved — nothing is lost.
            t = threading.Thread(
                target=_run_background_learning,
                args=(req.patient_id, draft_text),
                daemon=True
            )
            t.start()

        except Exception as e:
            _jobs[job_id] = {"status": "error", "error": str(e)}

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "running"}


@app.get("/run/{job_id}")
def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@app.get("/summary/{patient_id}")
def get_summary(patient_id: str):
    txt = OUTPUT_DIR / f"{patient_id.lower()}_summary.txt"
    jsn = OUTPUT_DIR / f"{patient_id.lower()}_summary.json"
    if not txt.exists():
        raise HTTPException(404, f"No summary for {patient_id}. Run POST /run first.")
    return {
        "patient_id":   patient_id,
        "summary_text": txt.read_text(encoding="utf-8"),
        "summary_json": json.loads(jsn.read_text(encoding="utf-8")) if jsn.exists() else {}
    }


@app.get("/trace/{patient_id}")
def get_trace(patient_id: str):
    trace = OUTPUT_DIR / f"{patient_id.lower()}_trace.txt"
    if not trace.exists():
        raise HTTPException(404, "No trace found")
    return {"patient_id": patient_id, "trace": trace.read_text(encoding="utf-8")}


# ── Learning endpoints ─────────────────────────────────────────

@app.get("/learning/status")
def learning_status():
    """Real-time learning state + live Pinecone count. Poll this to show live activity."""
    return {
        "running":           _learning_state["running"],
        "patient_id":        _learning_state["patient_id"],
        "stage":             _learning_state["stage"],
        "started_at":        _learning_state["started_at"],
        "corrections_total": get_correction_count(),
    }


@app.get("/learning/events")
def learning_events():
    """Recent learning activity — one event per agent run. Newest first."""
    events = _load_events()
    return {
        "total_events":      len(events),
        "corrections_total": get_correction_count(),
        "events":            events[:20],   # last 20
    }


@app.get("/learning/corrections")
def get_corrections():
    return {"total": get_correction_count(), "corrections": get_all_corrections(50)}


@app.get("/learning/metrics")
def get_metrics():
    p = OUTPUT_DIR / "learning_metrics.json"
    if not p.exists():
        raise HTTPException(404, "No batch metrics yet.")
    return json.loads(p.read_text())


@app.get("/learning/improvement")
def get_improvement():
    p = OUTPUT_DIR / "improvement_summary.json"
    if not p.exists():
        raise HTTPException(404, "No improvement summary yet.")
    return json.loads(p.read_text())


@app.post("/learning/run")
def trigger_batch_learning(background_tasks: BackgroundTasks):
    """Run the full multi-iteration batch learning loop (slow — use for offline training)."""
    from learning.run_loop import run_full_learning_loop
    background_tasks.add_task(run_full_learning_loop)
    return {"status": "started", "message": "Batch learning loop started. Check /learning/metrics when done."}
