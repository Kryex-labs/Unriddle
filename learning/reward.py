"""
Reward / Accuracy Signal.

Measures how much a clinician had to edit the agent's draft.
Less editing = higher reward = agent is improving.

Metric: section-level normalized similarity (1 - edit_burden)
Range: 0.0 (completely rewritten) → 1.0 (no edits needed)
"""
import re
from difflib import SequenceMatcher


SECTIONS = [
    "DEMOGRAPHICS", "ADMISSION DATE", "DISCHARGE DATE",
    "PRINCIPAL DIAGNOSIS", "SECONDARY DX", "HOSPITAL COURSE",
    "PROCEDURES", "DISCHARGE MEDICATIONS", "ALLERGIES",
    "FOLLOW-UP", "PENDING RESULTS", "DISCHARGE CONDITION",
    "CLINICIAN FLAGS"
]


def _extract_section(text: str, section_name: str) -> str:
    """Extract content of a named section from the discharge summary text."""
    # Try to find section header and content until next header
    pattern = rf"{re.escape(section_name)}\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z ]+\s*[:\-]|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _similarity(a: str, b: str) -> float:
    """Normalized similarity between two strings (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compute_reward(draft: str, edited: str) -> dict:
    """
    Compute section-level and overall reward.
    Returns detailed metrics dict.
    """
    section_scores = {}
    for section in SECTIONS:
        draft_content  = _extract_section(draft, section)
        edited_content = _extract_section(edited, section)
        score = _similarity(draft_content, edited_content)
        section_scores[section] = {
            "similarity": round(score, 4),
            "edit_burden": round(1.0 - score, 4),
            "draft_len": len(draft_content),
            "edited_len": len(edited_content),
        }

    # Overall reward = mean section similarity (weighted: clinical sections matter more)
    weights = {
        "DISCHARGE MEDICATIONS": 2.0,
        "PRINCIPAL DIAGNOSIS":   2.0,
        "HOSPITAL COURSE":       1.5,
        "PENDING RESULTS":       1.5,
        "CLINICIAN FLAGS":       1.5,
        "FOLLOW-UP":             1.2,
    }
    weighted_sum   = sum(section_scores[s]["similarity"] * weights.get(s, 1.0) for s in SECTIONS)
    total_weight   = sum(weights.get(s, 1.0) for s in SECTIONS)
    overall_reward = round(weighted_sum / total_weight, 4)
    overall_edit   = round(1.0 - overall_reward, 4)

    return {
        "overall_reward":      overall_reward,
        "overall_edit_burden": overall_edit,
        "sections":            section_scores,
        "interpretation": (
            "Excellent" if overall_reward >= 0.90 else
            "Good"       if overall_reward >= 0.75 else
            "Fair"       if overall_reward >= 0.60 else
            "Needs improvement"
        )
    }


def compare_iterations(metrics_by_iter: list[dict]) -> dict:
    """Compute improvement across iterations."""
    if len(metrics_by_iter) < 2:
        return {"message": "Need at least 2 iterations to measure improvement"}

    baseline    = metrics_by_iter[0]["overall_reward"]
    final       = metrics_by_iter[-1]["overall_reward"]
    improvement = round(final - baseline, 4)
    pct_improve = round((improvement / max(baseline, 0.001)) * 100, 1)

    per_iter = [{"iteration": i, "reward": m["overall_reward"], "edit_burden": m["overall_edit_burden"]}
                for i, m in enumerate(metrics_by_iter)]

    return {
        "baseline_reward":   baseline,
        "final_reward":      final,
        "absolute_improvement": improvement,
        "percent_improvement":  pct_improve,
        "per_iteration":     per_iter,
        "verdict": f"Edit burden reduced by {round((1-metrics_by_iter[0]['overall_edit_burden'])/(max(1-metrics_by_iter[-1]['overall_edit_burden'],0.001))*100-100, 1)}% — agent learned from {len(metrics_by_iter)-1} doctor review(s)"
    }
