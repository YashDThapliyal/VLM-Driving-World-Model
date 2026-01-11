# src/temporal_smoothing.py

import csv
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any


INPUT_FILENAME = "world_state_claude.csv"
OUTPUT_FILENAME = "world_state_claude_smoothed.csv"


def majority_vote(values: List[str]) -> str:
    """Return the most common non-empty value, or '' if none."""
    non_empty = [v for v in values if v]
    if not non_empty:
        return ""
    counts = Counter(non_empty)
    return counts.most_common(1)[0][0]


def smooth_sequence(rows: List[Dict[str, Any]], window: int = 3) -> List[Dict[str, Any]]:
    """
    Apply temporal smoothing over a list of rows from a single clip,
    assuming they are sorted by frame_index.
    """
    n = len(rows)
    half = window // 2

    smoothed: List[Dict[str, Any]] = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n - 1, i + half)

        window_rows = rows[start : end + 1]

        aff_vals = [r["affordance"] for r in window_rows]
        y_vals = [r["yield_to"] for r in window_rows]
        lead_vals = [r["lead_state"] for r in window_rows]

        aff_s = majority_vote(aff_vals)
        y_s = majority_vote(y_vals)
        lead_s = majority_vote(lead_vals)

        row = dict(rows[i])  # copy original row
        row["affordance_smoothed"] = aff_s or row["affordance"]
        row["yield_to_smoothed"] = y_s or row["yield_to"]
        row["lead_state_smoothed"] = lead_s or row["lead_state"]
        smoothed.append(row)

    return smoothed


def main():
    project_root = Path(__file__).resolve().parents[1]
    preds_root = project_root / "data" / "predictions"

    in_path = preds_root / INPUT_FILENAME
    out_path = preds_root / OUTPUT_FILENAME

    if not in_path.exists():
        raise FileNotFoundError(f"Input predictions file not found: {in_path}")

    # Read all rows
    with in_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("No rows in input predictions file.")

    # Normalize frame_index to int
    for r in rows:
        r["frame_index"] = int(r["frame_index"])

    # Group by clip
    by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        clip = r["clip"]
        by_clip.setdefault(clip, []).append(r)

    # Sort each clip by frame_index and smooth
    all_smoothed: List[Dict[str, Any]] = []
    for clip, clip_rows in by_clip.items():
        clip_rows_sorted = sorted(clip_rows, key=lambda r: r["frame_index"])
        smoothed_rows = smooth_sequence(clip_rows_sorted, window=3)
        all_smoothed.extend(smoothed_rows)

    # Preserve original fieldnames and add smoothed columns
    base_fields = list(rows[0].keys())
    extra_fields = ["affordance_smoothed", "yield_to_smoothed", "lead_state_smoothed"]
    fieldnames = base_fields + extra_fields

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_smoothed:
            writer.writerow(r)

    print(f"Saved smoothed predictions to {out_path}")


if __name__ == "__main__":
    main()
