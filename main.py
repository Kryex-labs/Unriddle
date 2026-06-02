"""
Entry point — run the discharge summary agent on all patients.
Usage: python main.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from discharge_agent.agent import DischargeAgent
from project_paths import OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ["PATIENT_1", "PATIENT_2"]

def main():
    agent = DischargeAgent()

    for patient_id in PATIENTS:
        print(f"\n{'='*60}")
        print(f"Processing {patient_id} with Claude Opus 4.5...")
        print(f"{'='*60}")

        trace_path = str(OUTPUT_DIR / f"{patient_id.lower()}_trace.txt")
        summary_json_path = OUTPUT_DIR / f"{patient_id.lower()}_summary.json"
        summary_txt_path = OUTPUT_DIR / f"{patient_id.lower()}_summary.txt"

        try:
            summary = agent.run(patient_id=patient_id, trace_path=trace_path)

            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary.model_dump(), f, indent=2, default=str)
            print(f"\nJSON saved: {summary_json_path}")

            readable = summary.to_readable()
            with open(summary_txt_path, "w", encoding="utf-8") as f:
                f.write(readable)
            print(f"TXT  saved: {summary_txt_path}")
            print(f"Trace saved: {trace_path}")
            print("\n" + readable)

        except Exception as e:
            import traceback
            print(f"FATAL ERROR for {patient_id}: {e}")
            traceback.print_exc()
            with open(OUTPUT_DIR / f"{patient_id.lower()}_error.txt", "w", encoding="utf-8") as f:
                f.write(f"AGENT FAILED\n{traceback.format_exc()}")

    print("\n\nAll patients processed. Check output/ directory.")

if __name__ == "__main__":
    main()
