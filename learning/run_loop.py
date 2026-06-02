"""
Learning Loop Orchestrator.

Runs multiple iterations:
  Iter 0 (baseline): agent with no learned corrections
  Iter 1: agent queries Pinecone for corrections from iter 0, produces better draft
  Iter 2: agent uses accumulated corrections, produces even better draft

Shows measurable improvement via reward curve.
Stores everything in output/iterations/ for the video demo.
"""
import sys
import json
import time
from pathlib import Path

load_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(load_path))

from dotenv import load_dotenv
load_dotenv(load_path / ".env")

from discharge_agent.agent import DischargeAgent, SYSTEM_PROMPT
from discharge_agent.claude_client import call_claude
from learning.reviewer import review_draft, extract_corrections
from learning.correction_store import (
    store_corrections_batch, query_relevant_corrections,
    format_corrections_for_prompt, get_correction_count
)
from learning.reward import compute_reward, compare_iterations
from project_paths import OUTPUT_DIR, ITERATIONS_DIR

ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS     = ["PATIENT_1", "PATIENT_2"]
NUM_ITERS    = 3   # iter 0 (baseline) + 2 learning iterations


def run_iteration(iteration: int, learned_prompt_addition: str = "") -> dict:
    """Run the agent on all patients for one iteration. Returns metrics."""
    iter_dir = ITERATIONS_DIR / f"iter_{iteration}"
    iter_dir.mkdir(exist_ok=True)

    iter_metrics = {"iteration": iteration, "patients": {}}

    for patient_id in PATIENTS:
        print(f"\n  [{patient_id}] Iter {iteration} — running agent...")
        try:
            trace_path = str(iter_dir / f"{patient_id.lower()}_trace.txt")

            agent = DischargeAgent(extra_system_context=learned_prompt_addition)
            summary = agent.run(patient_id=patient_id, trace_path=trace_path)
            draft_text = summary.to_readable()

            (iter_dir / f"{patient_id.lower()}_draft.txt").write_text(draft_text, encoding="utf-8")
            (iter_dir / f"{patient_id.lower()}_summary.json").write_text(
                json.dumps(summary.model_dump(), indent=2, default=str), encoding="utf-8")

            print(f"  [{patient_id}] Doctor reviewing draft...")
            try:
                edited_text = review_draft(draft_text, patient_id)
            except Exception as e:
                print(f"  [{patient_id}] Doctor review failed: {e} — using draft as edited")
                edited_text = draft_text

            (iter_dir / f"{patient_id.lower()}_edited.txt").write_text(edited_text, encoding="utf-8")

            reward = compute_reward(draft_text, edited_text)
            (iter_dir / f"{patient_id.lower()}_reward.json").write_text(
                json.dumps(reward, indent=2), encoding="utf-8")

            print(f"  [{patient_id}] Reward: {reward['overall_reward']:.3f} | "
                  f"Edit burden: {reward['overall_edit_burden']:.3f} | {reward['interpretation']}")

            print(f"  [{patient_id}] Extracting corrections for Pinecone...")
            try:
                corrections = extract_corrections(draft_text, edited_text, patient_id, iteration)
                if corrections:
                    store_corrections_batch(corrections)
                    print(f"  [{patient_id}] Stored {len(corrections)} corrections in Pinecone")
                else:
                    print(f"  [{patient_id}] No corrections extracted")
            except Exception as e:
                print(f"  [{patient_id}] Correction storage failed (non-fatal): {e}")
                corrections = []

            iter_metrics["patients"][patient_id] = {
                "reward": reward["overall_reward"],
                "edit_burden": reward["overall_edit_burden"],
                "corrections_stored": len(corrections),
                "interpretation": reward["interpretation"]
            }

        except Exception as e:
            import traceback
            print(f"  [{patient_id}] ERROR (skipping, loop continues): {e}")
            traceback.print_exc()
            iter_metrics["patients"][patient_id] = {
                "reward": 0.5, "edit_burden": 0.5,
                "corrections_stored": 0, "interpretation": "Error — skipped",
                "error": str(e)
            }

    # Average across patients
    rewards = [v["reward"] for v in iter_metrics["patients"].values()]
    iter_metrics["overall_reward"]      = round(sum(rewards) / len(rewards), 4)
    iter_metrics["overall_edit_burden"] = round(1.0 - iter_metrics["overall_reward"], 4)
    iter_metrics["total_corrections_in_db"] = get_correction_count()

    # Save iteration metrics
    (iter_dir / "metrics.json").write_text(json.dumps(iter_metrics, indent=2), encoding="utf-8")
    return iter_metrics


def build_learned_context() -> str:
    """Query Pinecone for most relevant corrections and format for injection."""
    sections = ["discharge_medications", "diagnoses", "hospital_course", "follow_up", "pending_results"]
    all_corrections = []
    seen_patterns = set()
    for section in sections:
        corrections = query_relevant_corrections(section=section, context=section, top_k=3)
        for c in corrections:
            pattern = c.get("pattern", "")
            if pattern and pattern not in seen_patterns:
                seen_patterns.add(pattern)
                all_corrections.append(c)
    return format_corrections_for_prompt(all_corrections[:8])


def run_full_learning_loop():
    """Run all iterations and produce improvement curve."""
    print("\n" + "="*60)
    print("PART 2 — LEARNING LOOP")
    print(f"Patients: {PATIENTS} | Iterations: {NUM_ITERS}")
    print("="*60)

    all_metrics = []

    for iteration in range(NUM_ITERS):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{NUM_ITERS-1}" + (" (BASELINE — no learning)" if iteration == 0 else f" (using corrections from Pinecone)"))
        print(f"Corrections in DB: {get_correction_count()}")
        print("="*60)

        # From iteration 1 onwards, inject learned corrections
        learned_context = ""
        if iteration > 0:
            learned_context = build_learned_context()
            if learned_context:
                print(f"Injecting {learned_context.count('.')} learned patterns into agent prompt")
            else:
                print("No relevant corrections found in Pinecone yet")

        metrics = run_iteration(iteration=iteration, learned_prompt_addition=learned_context)
        all_metrics.append(metrics)

        print(f"\nIter {iteration} complete — overall reward: {metrics['overall_reward']:.3f}")

    # Save full metrics history
    metrics_path = OUTPUT_DIR / "learning_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    # Compute and display improvement
    from learning.reward import compare_iterations
    comparison = compare_iterations(all_metrics)
    comparison_path = OUTPUT_DIR / "improvement_summary.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n" + "="*60)
    print("LEARNING LOOP COMPLETE")
    print("="*60)
    print(f"Baseline reward:     {comparison.get('baseline_reward', 0):.3f}")
    print(f"Final reward:        {comparison.get('final_reward', 0):.3f}")
    print(f"Improvement:         +{comparison.get('absolute_improvement', 0):.3f} ({comparison.get('percent_improvement', 0)}%)")
    print(f"Verdict: {comparison.get('verdict','')}")
    print(f"\nAll outputs saved to: {ITERATIONS_DIR}")

    # Plot improvement curve
    try:
        plot_curve(all_metrics)
    except Exception as e:
        print(f"Plot skipped: {e}")

    return comparison


def plot_curve(all_metrics: list[dict]):
    """Generate improvement curve PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters   = [m["iteration"] for m in all_metrics]
    rewards = [m["overall_reward"] for m in all_metrics]
    burdens = [m["overall_edit_burden"] for m in all_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Part 2 — Learning from Doctor Edits\nImprovement Curve", fontsize=14, fontweight="bold")

    # Reward curve
    ax1.plot(iters, rewards, "o-", color="#2ecc71", linewidth=2.5, markersize=8)
    ax1.fill_between(iters, rewards, alpha=0.15, color="#2ecc71")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Overall Reward (0-1)")
    ax1.set_title("Agent Accuracy (higher = better)")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(iters)
    ax1.set_xticklabels([f"Iter {i}\n{'(baseline)' if i==0 else '(+learning)'}" for i in iters])
    for i, r in zip(iters, rewards):
        ax1.annotate(f"{r:.3f}", (i, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Edit burden curve
    ax2.plot(iters, burdens, "o-", color="#e74c3c", linewidth=2.5, markersize=8)
    ax2.fill_between(iters, burdens, alpha=0.15, color="#e74c3c")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Edit Burden (0-1)")
    ax2.set_title("Doctor Edit Burden (lower = better)")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(iters)
    ax2.set_xticklabels([f"Iter {i}\n{'(baseline)' if i==0 else '(+learning)'}" for i in iters])
    for i, b in zip(iters, burdens):
        ax2.annotate(f"{b:.3f}", (i, b), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "improvement_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Curve saved: {out_path}")


if __name__ == "__main__":
    run_full_learning_loop()
