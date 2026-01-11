# src/build_state_and_planner.py

from __future__ import annotations
from pathlib import Path
import csv
from typing import Dict, Tuple, List, Any, Optional


Key = Tuple[str, int]  # (clip, frame_index)


def load_raw_world(path: Path) -> Dict[Key, Dict[str, str]]:
    """
    Load raw VLM world state (no smoothing).
    Expected columns: clip, frame_index, affordance, yield_to, lead_state, ...
    """
    raw: Dict[Key, Dict[str, str]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"].strip()
            fi = int(row["frame_index"])
            key = (clip, fi)
            raw[key] = {
                "affordance": row["affordance"].strip(),
                "yield_to": row["yield_to"].strip(),
                "lead_state": row["lead_state"].strip(),
            }
    return raw


def load_smoothed_world(path: Path) -> List[Dict[str, Any]]:
    """
    Load smoothed world state rows.
    Expected columns include:
      clip, frame_index, affordance_smoothed, yield_to_smoothed, lead_state_smoothed
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame_index"] = int(row["frame_index"])
            rows.append(row)
    return rows


def reinject_go_short(
    raw_map: Dict[Key, Dict[str, str]],
    smoothed_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each frame, if raw predicted affordance == 'go' but smoothed did not,
    and we're not yielding to pedestrians, overwrite the smoothed triple
    with the raw triple (SHORT mode: no temporal stretching).
    """
    updated: List[Dict[str, Any]] = []

    for row in smoothed_rows:
        clip = row["clip"].strip()
        fi = int(row["frame_index"])
        key = (clip, fi)

        # Base smoothed triple
        a_sm = row.get("affordance_smoothed", row.get("affordance", "")).strip()
        y_sm = row.get("yield_to_smoothed", row.get("yield_to", "")).strip()
        l_sm = row.get("lead_state_smoothed", row.get("lead_state", "")).strip()

        a_final, y_final, l_final = a_sm, y_sm, l_sm

        raw = raw_map.get(key)
        if raw is not None:
            a_raw = raw["affordance"]
            y_raw = raw["yield_to"]
            l_raw = raw["lead_state"]

            # Re-inject GO only if:
            #  - raw saw GO
            #  - smoothed did NOT keep GO
            #  - we're not dealing with pedestrians (they dominate)
            #  - raw is not yielding to ped either
            if (
                a_raw == "go"
                and a_sm != "go"
                and y_sm != "ped"
                and y_raw != "ped"
            ):
                a_final, y_final, l_final = a_raw, y_raw, l_raw

        row["affordance_final"] = a_final
        row["yield_to_final"] = y_final
        row["lead_state_final"] = l_final
        updated.append(row)

    return updated


def map_world_to_state(a: str, y: str, l: str) -> str:
    """
    Map (affordance, yield_to, lead_state) to a symbolic driving state.
    This is the mid-stack state machine abstraction.
    """
    a = a.strip()
    y = y.strip()
    l = l.strip()

    # Pedestrians dominate
    if y == "ped":
        if a == "stop":
            return "STOP_PED"
        else:
            return "YIELD_PED"

    # Lead-based constraints
    if y == "lead":
        if l == "stopped":
            if a == "stop":
                return "STOP_LEAD"
            else:
                return "FOLLOW_QUEUE"
        else:
            # lead moving or unknown
            if a == "go":
                return "GO_FOLLOW"
            else:
                return "FOLLOW_QUEUE"

    # No explicit yield target
    if a == "go" and y == "none":
        if l in ("none", ""):
            return "GO_FREE"
        else:
            return "GO"

    if a == "wait" and y == "none":
        return "NEGOTIATE"

    # default catch-all
    return "UNKNOWN"


def segment_states(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Given per-frame rows (with *_final fields), build segments of constant state.
    Returns segment list with:
      clip, segment_id, start_frame, end_frame, state,
      affordance, yield_to, lead_state
    """
    by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)

    segments: List[Dict[str, Any]] = []

    for clip, clip_rows in by_clip.items():
        clip_rows = sorted(clip_rows, key=lambda r: r["frame_index"])

        current_state: Optional[str] = None
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        current_aff: str = ""
        current_y: str = ""
        current_l: str = ""

        seg_id = 0

        for r in clip_rows:
            fi = int(r["frame_index"])
            a = r["affordance_final"]
            y = r["yield_to_final"]
            l = r["lead_state_final"]
            s = map_world_to_state(a, y, l)

            if current_state is None:
                current_state = s
                current_start = fi
                current_end = fi
                current_aff, current_y, current_l = a, y, l
            else:
                if s == current_state and fi == current_end + 1:
                    current_end = fi
                    # keep first triple as canonical
                else:
                    # close previous segment
                    segments.append(
                        {
                            "clip": clip,
                            "segment_id": seg_id,
                            "start_frame": current_start,
                            "end_frame": current_end,
                            "state": current_state,
                            "affordance": current_aff,
                            "yield_to": current_y,
                            "lead_state": current_l,
                        }
                    )
                    seg_id += 1
                    # start new
                    current_state = s
                    current_start = fi
                    current_end = fi
                    current_aff, current_y, current_l = a, y, l

        # close last
        if current_state is not None:
            segments.append(
                {
                    "clip": clip,
                    "segment_id": seg_id,
                    "start_frame": current_start,
                    "end_frame": current_end,
                    "state": current_state,
                    "affordance": current_aff,
                    "yield_to": current_y,
                    "lead_state": current_l,
                }
            )

    return segments


def add_prev_next(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each clip, add prev_state and next_state to segments.
    """
    by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for seg in segments:
        by_clip.setdefault(seg["clip"], []).append(seg)

    out: List[Dict[str, Any]] = []

    for clip, clip_segs in by_clip.items():
        clip_segs = sorted(clip_segs, key=lambda s: s["segment_id"])
        n = len(clip_segs)
        for i, seg in enumerate(clip_segs):
            prev_state = clip_segs[i - 1]["state"] if i > 0 else ""
            next_state = clip_segs[i + 1]["state"] if i < n - 1 else ""
            seg["prev_state"] = prev_state
            seg["next_state"] = next_state
            out.append(seg)

    return out


def map_state_to_planner(state: str) -> Dict[str, str]:
    """
    Map a symbolic state to a planner command abstraction.
    """
    state = state.strip()

    if state == "FOLLOW_QUEUE":
        return {
            "planner_cmd": "FOLLOW",
            "reason": "traffic_queue",
            "target": "lead",
            "until": "lead_moves",
        }
    if state == "GO_FOLLOW":
        return {
            "planner_cmd": "GO",
            "reason": "lead_moving",
            "target": "lead",
            "until": "constraint_reappears",
        }
    if state in ("GO_FREE", "GO"):
        return {
            "planner_cmd": "GO",
            "reason": "corridor_clear",
            "target": "",
            "until": "constraint_reappears",
        }
    if state == "YIELD_PED":
        return {
            "planner_cmd": "WAIT",
            "reason": "pedestrian_in_path",
            "target": "ped",
            "until": "ped_clears",
        }
    if state == "STOP_PED":
        return {
            "planner_cmd": "STOP",
            "reason": "pedestrian_blocking",
            "target": "ped",
            "until": "ped_clears",
        }
    if state == "STOP_LEAD":
        return {
            "planner_cmd": "STOP",
            "reason": "lead_vehicle_stopped",
            "target": "lead",
            "until": "lead_moves",
        }
    if state == "NEGOTIATE":
        return {
            "planner_cmd": "WAIT",
            "reason": "negotiating_gap",
            "target": "",
            "until": "scene_resolves",
        }

    # Fallback
    return {
        "planner_cmd": "WAIT",
        "reason": "unknown_state",
        "target": "",
        "until": "state_resolved",
    }


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    project_root = Path(__file__).resolve().parents[1]
    preds_root = project_root / "data" / "predictions"

    raw_path = preds_root / "world_state_claude.csv"
    smoothed_path = preds_root / "world_state_claude_smoothed.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw world state not found: {raw_path}")
    if not smoothed_path.exists():
        raise FileNotFoundError(f"Smoothed world state not found: {smoothed_path}")

    print(f"Loading raw from: {raw_path}")
    print(f"Loading smoothed from: {smoothed_path}")

    raw_map = load_raw_world(raw_path)
    smoothed_rows = load_smoothed_world(smoothed_path)

    # 1) Re-inject short GO bursts
    final_rows = reinject_go_short(raw_map, smoothed_rows)

    final_path = preds_root / "world_state_final.csv"
    # save final frame-level world state
    if final_rows:
        fieldnames = list(final_rows[0].keys())
        write_csv(final_path, fieldnames, final_rows)
        print(f"Saved final frame-level world state (with GO reinjected) to {final_path}")

    # 2) Build symbolic state machine segments
    segs = segment_states(final_rows)
    segs = add_prev_next(segs)

    machine_path = preds_root / "world_state_machine.csv"
    if segs:
        machine_fields = [
            "clip",
            "segment_id",
            "start_frame",
            "end_frame",
            "state",
            "prev_state",
            "next_state",
            "affordance",
            "yield_to",
            "lead_state",
        ]
        write_csv(machine_path, machine_fields, segs)
        print(f"Saved symbolic state machine to {machine_path}")

    # 3) Build planner abstraction
    planner_rows: List[Dict[str, Any]] = []
    for seg in segs:
        st = seg["state"]
        planner_info = map_state_to_planner(st)
        r = dict(seg)
        r.update(planner_info)
        planner_rows.append(r)

    planner_path = preds_root / "planner_commands.csv"
    if planner_rows:
        planner_fields = [
            "clip",
            "segment_id",
            "start_frame",
            "end_frame",
            "state",
            "prev_state",
            "next_state",
            "planner_cmd",
            "reason",
            "target",
            "until",
        ]
        write_csv(planner_path, planner_fields, planner_rows)
        print(f"Saved planner command sequence to {planner_path}")


if __name__ == "__main__":
    main()
