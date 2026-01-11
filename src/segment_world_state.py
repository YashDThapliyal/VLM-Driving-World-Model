# src/segment_world_state.py

from pathlib import Path
import csv
from typing import List, Dict, Any, Tuple


def load_smoothed_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["frame_index"] = int(r["frame_index"])
    return rows


def infer_phase(seg) -> str:
    """
    Deterministic micro-planner heuristic for human readable annotation.
    This can be replaced later by a VLM narrative (phase B).
    """
    a = seg["affordance"]
    y = seg["yield_to"]
    l = seg["lead_state"]

    if a == "stop":
        if y == "ped":
            return "STOP for pedestrian"
        if y == "lead":
            return "STOP behind vehicle"
        return "STOP"
    if a == "wait":
        if y == "ped":
            return "YIELD to pedestrian"
        if y == "lead":
            return "FOLLOW / queue"
        return "WAIT"
    if a == "go":
        if y == "lead" and l == "moving":
            return "FOLLOW moving lead"
        return "GO"
    return "UNKNOWN"


def segment_clip(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    def key(r: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            r["affordance_smoothed"],
            r["yield_to_smoothed"],
            r["lead_state_smoothed"],
        )

    segments: List[Dict[str, Any]] = []

    current_start = rows[0]["frame_index"]
    current_end = rows[0]["frame_index"]
    current_key = key(rows[0])

    for r in rows[1:]:
        k = key(r)
        fi = r["frame_index"]
        # same semantic + contiguous frame
        if k == current_key and fi == current_end + 1:
            current_end = fi
        else:
            segments.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "affordance": current_key[0],
                    "yield_to": current_key[1],
                    "lead_state": current_key[2],
                }
            )
            current_start = fi
            current_end = fi
            current_key = k

    segments.append(
        {
            "start": current_start,
            "end": current_end,
            "affordance": current_key[0],
            "yield_to": current_key[1],
            "lead_state": current_key[2],
        }
    )

    return segments


def main():
    project_root = Path(__file__).resolve().parents[1]
    preds_root = project_root / "data" / "predictions"
    csv_path = preds_root / "world_state_claude_smoothed.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Smoothed predictions file not found: {csv_path}")

    rows = load_smoothed_rows(csv_path)

    # group rows per clip
    by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)

    # CSV output
    out_path = preds_root / "world_state_segments.csv"
    fieldnames = [
        "clip",
        "segment_id",
        "start",
        "end",
        "affordance",
        "yield_to",
        "lead_state",
        "phase"
    ]
    with out_path.open("w", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()

        for clip, clip_rows in by_clip.items():
            clip_rows = sorted(clip_rows, key=lambda r: r["frame_index"])
            segments = segment_clip(clip_rows)

            print(f"\n=== Clip: {clip} ===")
            for seg_id, seg in enumerate(segments):
                seg["phase"] = infer_phase(seg)

                # debug printing
                s = seg["start"]
                e = seg["end"]
                a = seg["affordance"]
                y = seg["yield_to"]
                l = seg["lead_state"]
                phase = seg["phase"]

                if s == e:
                    span = f"frame {s}"
                else:
                    span = f"frames {s}–{e}"

                print(f"{span}: {phase} ({a},{y},{l})")

                writer.writerow({
                    "clip": clip,
                    "segment_id": seg_id,  # resets per clip
                    "start": s,
                    "end": e,
                    "affordance": a,
                    "yield_to": y,
                    "lead_state": l,
                    "phase": phase,
                })

    print(f"\nSaved segments to {out_path}")


if __name__ == "__main__":
    main()
