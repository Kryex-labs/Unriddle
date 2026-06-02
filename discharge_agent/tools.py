"""
Agent tools — called by Claude Opus 4.5 during the agent loop.
Each returns a dict with 'status'. Agent decides when and which to call.
"""
import json
import time
import random
from pathlib import Path
from project_paths import OUTPUT_DIR, PATIENT_DATA_JSON

_DATA_PATH = PATIENT_DATA_JSON
_patient_pages: dict = {}

def _load_data():
    global _patient_pages
    if _patient_pages:
        return
    with open(_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    _patient_pages = {p["page"]: p["text"] for p in data["pages"]}

_DRUG_INTERACTIONS = {
    frozenset(["metformin", "contrast"]): "Metformin + iodinated contrast: risk of lactic acidosis — hold metformin 48h",
    frozenset(["warfarin", "aspirin"]): "Warfarin + Aspirin: increased bleeding risk — monitor INR closely",
    frozenset(["metronidazole", "alcohol"]): "Metronidazole + alcohol: disulfiram-like reaction",
    frozenset(["ciprofloxacin", "antacid"]): "Ciprofloxacin + antacid: reduced absorption — space by 2h",
    frozenset(["metformin", "alcohol"]): "Metformin + alcohol: increased lactic acidosis risk",
    frozenset(["insulin", "alcohol"]): "Insulin + alcohol: unpredictable hypoglycemia risk",
    frozenset(["nsaid", "metformin"]): "NSAIDs + Metformin: risk of renal impairment and lactic acidosis",
    frozenset(["ciprofloxacin", "metformin"]): "Ciprofloxacin + Metformin: may increase metformin levels — monitor renal function",
    frozenset(["insulin", "metformin"]): "Insulin + Metformin: monitor for hypoglycemia — dosing adjustment may be needed",
}


def list_patient_pages(patient_id: str) -> dict:
    """Get page index with doc type and preview. Call this FIRST."""
    _load_data()
    try:
        if patient_id == "PATIENT_1":
            pages = {pg: txt for pg, txt in _patient_pages.items() if pg <= 45}
        elif patient_id == "PATIENT_2":
            pages = {pg: txt for pg, txt in _patient_pages.items() if pg >= 46}
        else:
            return {"status": "error", "result": "Unknown patient_id. Use PATIENT_1 or PATIENT_2."}

        index = {}
        for pg, txt in sorted(pages.items()):
            if not txt.strip():
                continue
            preview = txt[:200].replace("\n", " ").strip()
            tl = txt.lower()
            if any(k in tl for k in ["admission record", "case record", "chief complaint"]):
                doc_type = "ADMISSION_NOTE"
            elif any(k in tl for k in ["consultation sheet", "consult"]):
                doc_type = "CONSULTATION_NOTE"
            elif any(k in tl for k in ["drug chart", "drugs in", "prescription", "regular prescription"]):
                doc_type = "MEDICATION_CHART"
            elif any(k in tl for k in ["investigation", "biochemistry", "pathology", "urine routine", "abg", "result value"]):
                doc_type = "LAB_RESULT"
            elif any(k in tl for k in ["nurses notes", "nursing documentation", "nursing assessment"]):
                doc_type = "NURSING_NOTE"
            elif any(k in tl for k in ["discharge check", "discharge summary", "discharge medication"]):
                doc_type = "DISCHARGE_DOC"
            elif any(k in tl for k in ["monitoring chart", "intake", "output chart", "graphic", "tpr", "icu-chart"]):
                doc_type = "MONITORING_CHART"
            elif any(k in tl for k in ["diagnosis:", "history:", "course", "past history"]):
                doc_type = "CLINICAL_NOTE"
            else:
                doc_type = "OTHER"
            index[str(pg)] = {"doc_type": doc_type, "preview": preview}

        return {"status": "ok", "page_index": index, "total_pages_with_text": len(index)}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def read_pages(patient_id: str, page_numbers: list) -> dict:
    """Read full text of specific pages. Max 5 per call."""
    _load_data()
    page_numbers = [int(p) for p in page_numbers[:5]]
    try:
        result = {}
        for pg in page_numbers:
            txt = _patient_pages.get(pg, "")
            result[str(pg)] = txt if txt.strip() else "[Page has no extractable text — scanned chart/form]"
        return {"status": "ok", "pages": result, "pages_returned": len(result)}
    except Exception as e:
        return {"status": "error", "result": str(e)}


def reconcile_medications(admission_meds: list, discharge_meds: list) -> dict:
    """
    Formally compare admission vs discharge medication lists.
    Returns added, stopped, changed, and unchanged medications.
    Call this after extracting both medication lists separately.
    Any change with no documented reason must be flagged.
    """
    if not admission_meds and not discharge_meds:
        return {"status": "error", "result": "Both medication lists are empty — cannot reconcile"}

    adm_names = {m.get("name", "").lower().strip(): m for m in admission_meds if m.get("name")}
    dis_names = {m.get("name", "").lower().strip(): m for m in discharge_meds if m.get("name")}

    added, stopped, changed, unchanged = [], [], [], []
    flags = []

    # Medications in discharge but not admission = ADDED
    for name, med in dis_names.items():
        if name not in adm_names:
            entry = {**med, "reconciliation_status": "ADDED"}
            if not med.get("change_reason"):
                entry["flag"] = "Added with no documented reason — requires clinician verification"
                flags.append(f"NEW medication '{med.get('name')}' has no documented reason for addition")
            added.append(entry)

    # Medications in admission but not discharge = STOPPED
    for name, med in adm_names.items():
        if name not in dis_names:
            entry = {**med, "reconciliation_status": "STOPPED"}
            entry["flag"] = "Stopped — no documentation found for discontinuation"
            flags.append(f"STOPPED medication '{med.get('name')}' — discontinuation not documented")
            stopped.append(entry)

    # Medications in both — check for dose/frequency changes
    for name in set(adm_names) & set(dis_names):
        adm = adm_names[name]
        dis = dis_names[name]
        adm_dose = adm.get("dose", "")
        dis_dose = dis.get("dose", "")
        if adm_dose and dis_dose and adm_dose.lower() != dis_dose.lower():
            entry = {**dis, "reconciliation_status": "CHANGED", "previous_dose": adm_dose}
            if not dis.get("change_reason"):
                entry["flag"] = f"Dose changed {adm_dose} → {dis_dose} with no documented reason"
                flags.append(f"CHANGED: '{dis.get('name')}' dose {adm_dose} → {dis_dose} — no reason documented")
            changed.append(entry)
        else:
            unchanged.append({**dis, "reconciliation_status": "UNCHANGED"})

    return {
        "status": "ok",
        "added": added,
        "stopped": stopped,
        "changed": changed,
        "unchanged": unchanged,
        "flags": flags,
        "summary": f"{len(added)} added, {len(stopped)} stopped, {len(changed)} changed, {len(unchanged)} unchanged"
    }


def detect_conflict(patient_id: str, field: str, source_a: str, value_a: str,
                    source_b: str, value_b: str) -> dict:
    """
    Detect and record a conflict between two source documents for the same field.
    Call this whenever two documents disagree on the same clinical fact.
    NEVER silently resolve conflicts — always surface them.
    """
    conflict = {
        "field": field,
        "conflict": f"CONFLICT: {source_a} states '{value_a}' but {source_b} states '{value_b}'",
        "source_a": source_a,
        "value_a": value_a,
        "source_b": source_b,
        "value_b": value_b,
        "resolution": "REQUIRES CLINICIAN REVIEW — do not auto-resolve"
    }

    # Persist conflict log
    log_path = OUTPUT_DIR / "conflicts.json"
    log_path.parent.mkdir(exist_ok=True)
    existing = []
    if log_path.exists():
        try:
            existing = json.load(open(log_path))
        except Exception:
            existing = []
    existing.append({"patient_id": patient_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **conflict})
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    return {"status": "ok", "conflict_recorded": conflict}


def check_drug_interaction(medications: list) -> dict:
    """Check for drug-drug interactions. Call after extracting full discharge med list."""
    if random.random() < 0.05:
        return {"status": "timeout", "result": "Drug interaction service timed out — retry or flag for manual pharmacist review"}

    meds_lower = [m.lower().strip() for m in medications]
    interactions = []
    for i, med1 in enumerate(meds_lower):
        for med2 in meds_lower[i+1:]:
            for known_key, warning in _DRUG_INTERACTIONS.items():
                km = list(known_key)
                if (km[0] in med1 or km[0] in med2) and (km[1] in med1 or km[1] in med2):
                    if warning not in interactions:
                        interactions.append(warning)

    return {
        "status": "ok",
        "interactions_found": len(interactions),
        "result": interactions if interactions else "No known interactions detected"
    }


def escalate_to_clinician(patient_id: str, field: str, reason: str) -> dict:
    """Escalate safety concern, conflict, or unexplained finding to clinician queue."""
    escalation = {
        "patient_id": patient_id, "field": field, "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "escalated"
    }
    log_path = OUTPUT_DIR / "escalations.json"
    log_path.parent.mkdir(exist_ok=True)
    existing = []
    if log_path.exists():
        try:
            existing = json.load(open(log_path, encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(escalation)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return {
        "status": "ok",
        "result": f"Escalated: [{field}] — {reason}",
        "escalation_id": f"{patient_id}_{field.replace(' ','_')}_{int(time.time())}"
    }


def flag_missing_field(patient_id: str, field_name: str, context: str = "") -> dict:
    """Flag a required field as MISSING or PENDING."""
    return {
        "status": "ok",
        "result": f"Field '{field_name}' marked MISSING/PENDING for clinician completion",
        "field": field_name, "context": context
    }


TOOL_SCHEMAS = [
    {
        "name": "list_patient_pages",
        "description": "Get page index with doc type and preview. ALWAYS call first.",
        "input_schema": {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
    },
    {
        "name": "read_pages",
        "description": "Read full text of specific pages (max 5). Use list_patient_pages first to identify relevant pages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "page_numbers": {"type": "array", "items": {"type": "integer"}, "description": "Max 5 page numbers"}
            },
            "required": ["patient_id", "page_numbers"]
        }
    },
    {
        "name": "reconcile_medications",
        "description": "Formally diff admission vs discharge medication lists. Returns added/stopped/changed/unchanged with flags for undocumented changes. Call this AFTER extracting both lists separately.",
        "input_schema": {
            "type": "object",
            "properties": {
                "admission_meds": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"name": {"type": "string"}, "dose": {"type": "string"}, "route": {"type": "string"}, "frequency": {"type": "string"}}},
                    "description": "Medications the patient was on at admission"
                },
                "discharge_meds": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"name": {"type": "string"}, "dose": {"type": "string"}, "route": {"type": "string"}, "frequency": {"type": "string"}, "change_reason": {"type": "string"}}},
                    "description": "Medications prescribed at discharge"
                }
            },
            "required": ["admission_meds", "discharge_meds"]
        }
    },
    {
        "name": "detect_conflict",
        "description": "Record a conflict when two documents disagree on the same clinical fact. NEVER silently resolve — always call this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "field": {"type": "string"},
                "source_a": {"type": "string", "description": "e.g. 'page 1 admission note'"},
                "value_a": {"type": "string"},
                "source_b": {"type": "string", "description": "e.g. 'page 48 consultation note'"},
                "value_b": {"type": "string"}
            },
            "required": ["patient_id", "field", "source_a", "value_a", "source_b", "value_b"]
        }
    },
    {
        "name": "check_drug_interaction",
        "description": "Check drug-drug interactions for the discharge medication list.",
        "input_schema": {
            "type": "object",
            "properties": {"medications": {"type": "array", "items": {"type": "string"}}},
            "required": ["medications"]
        }
    },
    {
        "name": "escalate_to_clinician",
        "description": "Escalate safety concern, conflict, or unexplained change to clinician review queue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "field": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["patient_id", "field", "reason"]
        }
    },
    {
        "name": "flag_missing_field",
        "description": "Flag required field as MISSING or PENDING when not found in documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "field_name": {"type": "string"},
                "context": {"type": "string"}
            },
            "required": ["patient_id", "field_name"]
        }
    }
]

TOOL_MAP = {
    "list_patient_pages": list_patient_pages,
    "read_pages": read_pages,
    "reconcile_medications": reconcile_medications,
    "detect_conflict": detect_conflict,
    "check_drug_interaction": check_drug_interaction,
    "escalate_to_clinician": escalate_to_clinician,
    "flag_missing_field": flag_missing_field,
}
