"""
Simulated Doctor Reviewer.

Claude Opus 4.5 plays a senior clinician with a CONSISTENT, HIDDEN editing policy.
The agent does not know this policy — it must learn it from accumulated corrections.
Produces (draft, edited) pairs for the learning loop.
"""
import json
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from discharge_agent.claude_client import call_claude

# The doctor's hidden editing policy — consistent across all reviews
# The agent does NOT see this. It must learn these patterns from corrections.
DOCTOR_SYSTEM = """You are a senior clinician reviewing an AI-generated discharge summary draft.

Apply these corrections CONSISTENTLY to every draft (the AI agent does not know these rules — it must learn them):

FORMATTING RULES:
1. MEDICATIONS: Always expand dosing abbreviations
   - 1-0-0 → "once daily (morning)"
   - 0-0-1 → "once daily (night)"
   - 1-0-1 → "twice daily (morning and night)"
   - BD → "twice daily"
   - TDS/TID → "three times daily"
   - SOS → "as needed (SOS)"
2. MEDICATIONS: Always include route of administration (oral / IV / subcutaneous / topical / inhaled)
3. MEDICATIONS: Format each as: "Drug (Brand) Dose Route Frequency — [STATUS]"
   Example: "Rabeprazole (Ractiper) 40mg oral once daily (morning) — NEW"

CLINICAL RULES:
4. DIAGNOSES: Format as "Condition — qualifier"
   Example: "Acute Gastroenteritis with Dehydration — resolved on discharge"
5. HOSPITAL COURSE: Must include admission vitals if mentioned anywhere in the summary
6. HOSPITAL COURSE: End with discharge disposition sentence
7. FOLLOW-UP: Always end with: "Return to ED immediately if symptoms worsen or new symptoms develop."
8. PENDING RESULTS: Always append: "Results to be communicated to patient by treating team/GP."
9. DISCHARGE CONDITION: Use one of: Stable / Improved / Stable with ongoing monitoring required
10. ALLERGIES: If NKDA — write "No Known Drug Allergies (NKDA) — verified"

SAFETY RULES:
11. Never remove any [MISSING], [PENDING], or [CONFLICT] flag — only add to them if needed
12. Never invent clinical facts — only improve format/style of existing content
13. If a medication flag exists, expand it with more specific guidance

Your output must be the COMPLETE edited discharge summary. Make all corrections according to the rules above.
Output ONLY the edited summary text — no commentary, no explanation."""


def review_draft(draft_text: str, patient_id: str) -> str:
    """
    Simulated doctor reviews the draft and returns an edited version.
    Applies the hidden editing policy consistently.
    """
    messages = [{
        "role": "user",
        "content": f"Please review and edit this discharge summary draft for {patient_id}:\n\n{draft_text}"
    }]
    response = call_claude(messages=messages, system=DOCTOR_SYSTEM, max_tokens=4096)
    edited = ""
    for block in response.get("content", []):
        if block.get("type") == "text":
            edited += block["text"]
    return edited.strip()


def extract_corrections(draft: str, edited: str, patient_id: str, iteration: int) -> list[dict]:
    """
    Use Claude to extract generalizable correction patterns from a (draft, edited) pair.
    Returns structured corrections ready to store in Pinecone.
    """
    extraction_system = """You are analyzing differences between an AI draft and a doctor-edited version.
Extract correction patterns that are GENERALIZABLE to future patients.
Output a JSON array only — no other text."""

    prompt = f"""Compare these two versions and extract correction patterns.

DRAFT:
{draft[:3000]}

DOCTOR-EDITED VERSION:
{edited[:3000]}

For each type of change made, output a JSON array:
[
  {{
    "section": "discharge_medications|diagnoses|hospital_course|follow_up|pending_results|discharge_condition|allergies",
    "agent_wrote": "exact text the agent wrote (short example)",
    "doctor_changed_to": "what the doctor changed it to (short example)",
    "pattern": "the generalizable rule being applied (one clear sentence)",
    "reward_delta": 0.1
  }}
]

Focus on patterns that would apply to OTHER patients too, not patient-specific facts.
Extract 3-8 most important corrections. Output JSON array only."""

    messages = [{"role": "user", "content": prompt}]
    response = call_claude(messages=messages, system=extraction_system, max_tokens=2048)

    text = ""
    for block in response.get("content", []):
        if block.get("type") == "text":
            text += block["text"]

    # Parse JSON
    try:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            corrections = json.loads(match.group())
            # Add metadata
            for c in corrections:
                c["patient_id"] = patient_id
                c["iteration"] = iteration
            return corrections
    except Exception as e:
        print(f"  Warning: could not parse corrections JSON: {e}")

    return []
