"""
Discharge Summary Agent — powered by Claude Opus 4.5.

Loop: Claude reasons -> calls tools -> gets results -> re-reasons -> ... -> produces final summary.
Hard step cap enforced. Full trace logged at every step.
No fabrication: every field is sourced or explicitly marked MISSING/PENDING/CONFLICT.
"""
import json
import time
from pathlib import Path
from .claude_client import call_claude
from .tools import TOOL_SCHEMAS, TOOL_MAP
from .models import DischargeSummary, ClinicalField, FieldStatus, Medication

MAX_STEPS = 15

SYSTEM_PROMPT = """You are a clinical AI assistant generating structured DRAFT discharge summaries for clinician review.

=== ABSOLUTE RULES — NEVER VIOLATE ===
1. NEVER invent, guess, or infer any clinical fact not explicitly stated in source documents.
2. If a field is not found: mark it [MISSING — requires clinician input].
3. If a lab result is not yet resulted: mark it [PENDING — result not yet available].
4. If two source documents contradict each other: call detect_conflict immediately and mark [CONFLICT].
5. If a medication was changed/added/stopped with NO documented reason: flag it and call escalate_to_clinician.
6. This output is ALWAYS a DRAFT — never auto-finalize.
7. If a drug interaction is found: always call escalate_to_clinician immediately.

=== YOUR WORKFLOW ===
Step 1. Call list_patient_pages to see all documents and their types.
Step 2. Read CLINICAL_NOTE and ADMISSION_NOTE pages first using read_pages (max 5 per call).
Step 3. Extract demographics, dates, diagnoses, hospital course, procedures from those pages.
Step 4. Read MEDICATION_CHART pages — extract ADMISSION medications as a separate explicit list.
Step 5. Read DISCHARGE_DOC pages — extract DISCHARGE medications as a separate explicit list.
Step 6. Call reconcile_medications(admission_meds, discharge_meds) — this gives you the formal diff with flags.
Step 7. Call check_drug_interaction with the discharge medication list.
Step 8. Read LAB_RESULT pages — note any pending results.
Step 9. Cross-check: if same field appears in multiple documents with different values, call detect_conflict.
Step 10. For every conflict or safety issue: call escalate_to_clinician.
Step 11. For every required field not found: call flag_missing_field.
Step 12. Output the final structured JSON summary.

=== REQUIRED FIELDS ===
demographics, admission_date, discharge_date, principal_diagnosis, secondary_diagnoses,
hospital_course, procedures, discharge_medications (with reconciliation vs admission),
allergies, follow_up, pending_results, discharge_condition

=== FINAL OUTPUT FORMAT ===
When you have gathered all available information, output ONLY this JSON (no other text):

```json
{
  "final_summary": {
    "patient_id": "...",
    "demographics": {"value": "...", "status": "FOUND", "source": "page X"},
    "admission_date": {"value": "...", "status": "FOUND", "source": "page X"},
    "discharge_date": {"value": "...", "status": "MISSING", "source": null},
    "principal_diagnosis": {"value": "...", "status": "FOUND", "source": "page X"},
    "secondary_diagnoses": {"value": "...", "status": "FOUND", "source": "page X"},
    "hospital_course": {"value": "...", "status": "FOUND", "source": "pages X, Y"},
    "procedures": {"value": "...", "status": "FOUND", "source": "page X"},
    "discharge_medications": [
      {
        "name": "Drug Name",
        "dose": "500mg",
        "route": "oral",
        "frequency": "BD",
        "status": "unchanged",
        "change_reason": null,
        "flag": null
      }
    ],
    "allergies": {"value": "...", "status": "FOUND", "source": "page X"},
    "follow_up": {"value": "...", "status": "MISSING", "source": null},
    "pending_results": {"value": "...", "status": "FOUND", "source": "page X"},
    "discharge_condition": {"value": "...", "status": "FOUND", "source": "page X"},
    "clinician_flags": [
      "List all escalations, conflicts, unexplained medication changes, and drug interactions here"
    ]
  }
}
```

For status use: FOUND, MISSING, PENDING, CONFLICT, or FLAGGED.
For discharge_medications status use: added, stopped, changed, unchanged."""


class DischargeAgent:
    def __init__(self, extra_system_context: str = ""):
        # extra_system_context: learned corrections injected from Pinecone
        self.extra_system_context = extra_system_context

    def run(self, patient_id: str, trace_path: str) -> DischargeSummary:
        messages = [
            {
                "role": "user",
                "content": f"Generate a discharge summary draft for {patient_id}. Follow your workflow exactly — start with list_patient_pages."
            }
        ]

        trace_lines = []
        step = 0
        final_summary = None
        flags_collected = []

        def log(msg: str):
            ts = time.strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            trace_lines.append(line)
            print(line)

        log(f"=== AGENT START: {patient_id} | Model: claude-opus-4-5 ===")

        while step < MAX_STEPS:
            step += 1
            log(f"\n--- STEP {step}/{MAX_STEPS} ---")

            effective_system = SYSTEM_PROMPT + (self.extra_system_context or "")
            try:
                response = call_claude(
                    messages=messages,
                    system=effective_system,
                    tools=TOOL_SCHEMAS,
                    max_tokens=4096
                )
            except Exception as e:
                log(f"LLM ERROR (fatal): {e}")
                break

            stop_reason = response.get("stop_reason", "")
            content_blocks = response.get("content", [])

            messages.append({"role": "assistant", "content": content_blocks})

            for block in content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    log(f"REASONING: {block['text'][:600]}")

            if stop_reason == "tool_use":
                tool_results = []
                for block in content_blocks:
                    if block.get("type") != "tool_use":
                        continue

                    tool_name = block["name"]
                    tool_input = block.get("input", {})
                    tool_use_id = block["id"]

                    log(f"TOOL CALL: {tool_name}({json.dumps(tool_input)[:300]})")
                    result = self._execute_tool(tool_name, tool_input, log)
                    log(f"TOOL RESULT: {json.dumps(result)[:400]}")

                    if tool_name == "escalate_to_clinician":
                        flags_collected.append(f"[ESCALATED] {tool_input.get('field')}: {tool_input.get('reason')}")
                    if tool_name == "flag_missing_field":
                        flags_collected.append(f"[MISSING] {tool_input.get('field_name')}: {tool_input.get('context','not found in documents')}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(result)
                    })

                messages.append({"role": "user", "content": tool_results})

            elif stop_reason == "end_turn":
                for block in content_blocks:
                    if block.get("type") == "text":
                        parsed = self._try_parse_final(block["text"], log)
                        if parsed:
                            existing_flags = parsed.get("clinician_flags", [])
                            parsed["clinician_flags"] = list(set(existing_flags + flags_collected))
                            final_summary = parsed
                            log("FINAL SUMMARY PARSED SUCCESSFULLY")
                            break

                if final_summary:
                    break

                log("No final JSON detected — prompting Claude to output structured summary")
                messages.append({
                    "role": "user",
                    "content": "You have gathered enough information. Now output the final discharge summary as JSON in the exact format specified (```json ... ```)."
                })
            else:
                log(f"Unexpected stop_reason: {stop_reason} — continuing")

        if step >= MAX_STEPS:
            log(f"HARD STEP CAP REACHED ({MAX_STEPS}) — saving best-effort summary")

        Path(trace_path).parent.mkdir(exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write("\n".join(trace_lines))
        log(f"Trace saved: {trace_path}")

        if final_summary:
            return self._build_summary(patient_id, final_summary, flags_collected)

        log("Building fallback empty summary — all fields MISSING")
        s = DischargeSummary(patient_id=patient_id)
        s.clinician_flags = flags_collected + ["AGENT DID NOT PRODUCE FINAL SUMMARY — full manual review required"]
        return s

    def _execute_tool(self, tool_name: str, tool_input: dict, log) -> dict:
        if tool_name not in TOOL_MAP:
            return {"status": "error", "result": f"Unknown tool: {tool_name}"}
        fn = TOOL_MAP[tool_name]
        for attempt in range(3):
            try:
                result = fn(**tool_input)
                if result.get("status") == "timeout" and attempt < 2:
                    log(f"  Timeout on {tool_name} (attempt {attempt+1}) — retrying")
                    time.sleep(1)
                    continue
                return result
            except Exception as e:
                if attempt == 2:
                    return {"status": "error", "result": f"{tool_name} failed after 3 attempts: {e}"}
                log(f"  Error in {tool_name} (attempt {attempt+1}): {e}")
                time.sleep(1)
        return {"status": "error", "result": "Tool failed"}

    def _try_parse_final(self, text: str, log) -> dict | None:
        import re
        for pattern in [
            r'```json\s*(.*?)\s*```',
            r'```\s*(\{.*?"final_summary".*?\})\s*```',
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if "final_summary" in data:
                        return data["final_summary"]
                except json.JSONDecodeError as e:
                    log(f"JSON parse error: {e}")
        try:
            data = json.loads(text.strip())
            if "final_summary" in data:
                return data["final_summary"]
        except Exception:
            pass
        return None

    def _build_summary(self, patient_id: str, data: dict, extra_flags: list) -> DischargeSummary:
        def _to_str(v) -> str:
            """Safely convert any value to string."""
            if v is None:
                return ""
            if isinstance(v, list):
                return "; ".join(str(i) for i in v if i)
            if isinstance(v, dict):
                return str(v.get("value") or v.get("text") or "")
            return str(v)

        def parse_field(raw) -> ClinicalField:
            try:
                if raw is None:
                    return ClinicalField(status=FieldStatus.MISSING)
                if isinstance(raw, list):
                    raw = "; ".join(str(i) for i in raw if i)
                if isinstance(raw, str):
                    su = raw.upper()
                    if "[MISSING" in su or raw.strip() == "":
                        return ClinicalField(status=FieldStatus.MISSING)
                    if "[PENDING" in su:
                        return ClinicalField(status=FieldStatus.PENDING, flag_reason=raw)
                    if "[CONFLICT" in su:
                        return ClinicalField(status=FieldStatus.CONFLICT, flag_reason=raw)
                    return ClinicalField(value=raw, status=FieldStatus.FOUND)
                if isinstance(raw, dict):
                    status_str = str(raw.get("status", "MISSING")).upper()
                    try:
                        status = FieldStatus(status_str)
                    except ValueError:
                        status = FieldStatus.MISSING
                    val = _to_str(raw.get("value"))
                    if not val or "[MISSING" in val.upper():
                        status = FieldStatus.MISSING
                        val = None
                    return ClinicalField(
                        value=val if val else None,
                        status=status,
                        source=_to_str(raw.get("source")) or None,
                        flag_reason=_to_str(raw.get("flag_reason") or raw.get("reason")) or None
                    )
                # Anything else — convert to string
                return ClinicalField(value=_to_str(raw), status=FieldStatus.FOUND)
            except Exception:
                return ClinicalField(status=FieldStatus.MISSING)

        meds = []
        for m in data.get("discharge_medications", []):
            if isinstance(m, dict) and m.get("name"):
                meds.append(Medication(
                    name=m.get("name", "Unknown"),
                    dose=m.get("dose"),
                    route=m.get("route"),
                    frequency=m.get("frequency"),
                    status=m.get("status", "unchanged"),
                    change_reason=m.get("change_reason"),
                    flag=m.get("flag")
                ))

        all_flags = list(set(data.get("clinician_flags", []) + extra_flags))

        return DischargeSummary(
            patient_id=patient_id,
            demographics=parse_field(data.get("demographics")),
            admission_date=parse_field(data.get("admission_date")),
            discharge_date=parse_field(data.get("discharge_date")),
            principal_diagnosis=parse_field(data.get("principal_diagnosis")),
            secondary_diagnoses=parse_field(data.get("secondary_diagnoses")),
            hospital_course=parse_field(data.get("hospital_course")),
            procedures=parse_field(data.get("procedures")),
            discharge_medications=meds,
            allergies=parse_field(data.get("allergies")),
            follow_up=parse_field(data.get("follow_up")),
            pending_results=parse_field(data.get("pending_results")),
            discharge_condition=parse_field(data.get("discharge_condition")),
            clinician_flags=all_flags
        )
