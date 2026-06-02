"""
FastAPI server — exposes the discharge agent and learning loop as REST endpoints.
Deployed on Railway. Frontend (Vercel) calls these endpoints.
"""
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from discharge_agent.agent import DischargeAgent
from learning.correction_store import (
    get_correction_count, get_all_corrections
)
from learning.run_loop import run_full_learning_loop, build_learned_context
from project_paths import OUTPUT_DIR

app = FastAPI(
    title="Discharge Summary Agent",
    description="Agentic AI for clinical discharge summaries with Pinecone-powered learning loop",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_jobs: dict = {}


class RunRequest(BaseModel):
    patient_id: str
    use_learned_corrections: bool = True


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "claude-opus-4-5",
        "corrections_in_pinecone": get_correction_count(),
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
            learned = build_learned_context() if req.use_learned_corrections else ""
            agent   = DischargeAgent(extra_system_context=learned)
            trace   = str(OUTPUT_DIR / f"{req.patient_id.lower()}_trace.txt")
            summary = agent.run(patient_id=req.patient_id, trace_path=trace)
            (OUTPUT_DIR / f"{req.patient_id.lower()}_summary.txt").write_text(summary.to_readable(), encoding="utf-8")
            (OUTPUT_DIR / f"{req.patient_id.lower()}_summary.json").write_text(
                json.dumps(summary.model_dump(), indent=2, default=str), encoding="utf-8")
            _jobs[job_id] = {"status": "done", "patient_id": req.patient_id}
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
    txt  = OUTPUT_DIR / f"{patient_id.lower()}_summary.txt"
    jsn  = OUTPUT_DIR / f"{patient_id.lower()}_summary.json"
    if not txt.exists():
        raise HTTPException(404, f"No summary for {patient_id}. Call POST /run first.")
    return {
        "patient_id":    patient_id,
        "summary_text":  txt.read_text(encoding="utf-8"),
        "summary_json":  json.loads(jsn.read_text(encoding="utf-8")) if jsn.exists() else {}
    }


@app.get("/trace/{patient_id}")
def get_trace(patient_id: str):
    trace = OUTPUT_DIR / f"{patient_id.lower()}_trace.txt"
    if not trace.exists():
        raise HTTPException(404, "No trace found")
    return {"patient_id": patient_id, "trace": trace.read_text(encoding="utf-8")}


@app.get("/learning/corrections")
def get_corrections():
    return {"total": get_correction_count(), "corrections": get_all_corrections(50)}


@app.get("/learning/metrics")
def get_metrics():
    p = OUTPUT_DIR / "learning_metrics.json"
    if not p.exists():
        raise HTTPException(404, "No metrics yet. Run POST /learning/run first.")
    return json.loads(p.read_text())


@app.get("/learning/improvement")
def get_improvement():
    p = OUTPUT_DIR / "improvement_summary.json"
    if not p.exists():
        raise HTTPException(404, "No improvement data yet.")
    return json.loads(p.read_text())


@app.post("/learning/run")
def trigger_learning(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_full_learning_loop)
    return {"status": "started", "message": "Learning loop running in background. Check /learning/metrics when done."}
