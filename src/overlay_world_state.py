# src/overlay_world_state.py

import argparse
from pathlib import Path
import csv
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def load_smoothed_preds(csv_path: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    pred_map: Dict[Tuple[str, int], Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            fi = int(row["frame_index"])
            key = (clip, fi)
            pred_map[key] = {
                "affordance": row.get("affordance_smoothed", row.get("affordance", "")),
                "yield_to": row.get("yield_to_smoothed", row.get("yield_to", "")),
                "lead_state": row.get("lead_state_smoothed", row.get("lead_state", "")),
            }
    return pred_map


def load_segments(segments_path: Path):
    by_clip = {}
    with segments_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            by_clip.setdefault(clip, []).append({
                "segment_id": int(row["segment_id"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "phase": row["phase"],
            })
    for clip in by_clip:
        by_clip[clip].sort(key=lambda s: s["start"])
    return by_clip


def load_gloss(gloss_path: Path):
    gmap = {}
    with gloss_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["clip"], int(row["segment_id"]))
            gmap[key] = {
                "gloss_short": row.get("gloss_short", "").strip(),
                "gloss_long": row.get("gloss_long", "").strip(),
            }
    return gmap


def load_planner(planner_path: Path):
    """
    Returns (clip, segment_id) -> planner dict:
    {
      behavior: FOLLOW / STOP / WAIT / YIELD (etc)
      intent:   FOLLOW(lead) until lead_moves
    }
    """
    pmap = {}
    with planner_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = row["clip"]
            seg_id = int(row["segment_id"])
            behavior = row.get("planner_cmd", "").strip().upper()

            target = row.get("target", "").strip()
            until = row.get("until", "").strip()

            if target and until:
                intent = f"{behavior.title()}({target}) until {until}"
            elif target:
                intent = f"{behavior.title()}({target})"
            else:
                intent = behavior.title()

            pmap[(clip, seg_id)] = {
                "behavior": behavior,
                "intent": intent,
            }
    return pmap


def find_segment(clip: str, frame_index: int, segments) -> Tuple[int, str]:
    segs = segments.get(clip, [])
    for seg in segs:
        if seg["start"] <= frame_index <= seg["end"]:
            return seg["segment_id"], seg["phase"]
    return -1, "UNKNOWN"


def collect_frame_paths(frames_dir: Path):
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.jpg or .png found in {frames_dir}")
    return paths


def draw_hud(img, clip_name, frame_index, state):
    if img.mode != "RGB":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    W, H = img.size

    base_font_size = max(18, min(40, H // 28))
    try:
        font = ImageFont.truetype("Menlo.ttf", base_font_size)
    except Exception:
        font = ImageFont.load_default()

    # HUD lines (clean cognitive order)
    lines = [
        f"CLIP: {clip_name}  FRAME: {frame_index}",
        f"PHASE: {state.get('phase','N/A')}",
    ]

    gs = state.get("gloss_short", "")
    if gs:
        lines.append(f"SCENE: {gs}")

    lines.append(f"AFFORDANCE: {state.get('affordance','N/A')}")
    lines.append(f"YIELD TO:   {state.get('yield_to','N/A')}")
    lines.append(f"LEAD STATE: {state.get('lead_state','N/A')}")

    behavior = state.get("behavior","")
    if behavior:
        lines.append(f"PLAN: {behavior}")

    intent = state.get("intent","")
    if intent:
        lines.append(f"INTENT: {intent}")

    def text_size(line):
        bbox = draw.textbbox((0, 0), line, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]

    text_w = 0
    text_h = 0
    for line in lines:
        w, h = text_size(line)
        text_w = max(text_w, w)
        text_h += h + 6

    box_w = text_w + 40
    box_h = text_h + 40
    box_coords = (10, 10, 10 + box_w, 10 + box_h)

    draw.rectangle(box_coords, fill=(0, 0, 0))
    y = box_coords[1] + 20
    x = box_coords[0] + 20
    for line in lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        _, h = text_size(line)
        y += h + 6

    return img


def make_overlay_video_for_clip(clip_name, frames_root, preds, segments, gloss, planner, out_dir, fps=2):
    frames_dir = frames_root / clip_name
    frame_paths = collect_frame_paths(frames_dir)

    frames = []
    for idx, frame_path in enumerate(frame_paths):
        frame_index = idx
        key = (clip_name, frame_index)

        base = preds.get(key, {"affordance":"N/A","yield_to":"N/A","lead_state":"N/A"})

        seg_id, phase = find_segment(clip_name, frame_index, segments)
        gloss_short = ""
        behavior = ""
        intent = ""

        if seg_id >= 0:
            g = gloss.get((clip_name, seg_id))
            if g:
                gloss_short = g.get("gloss_short", "")
            p = planner.get((clip_name, seg_id))
            if p:
                behavior = p.get("behavior","")
                intent = p.get("intent","")

        state = {
            **base,
            "phase": phase,
            "gloss_short": gloss_short,
            "behavior": behavior,
            "intent": intent,
        }

        img = Image.open(frame_path).convert("RGB")
        img = draw_hud(img, clip_name, frame_index, state)
        frames.append(img)

    frames_np = [np.array(im) for im in frames]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_name}_overlay.mp4"

    clip = ImageSequenceClip(frames_np, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264")
    print(f"Saved overlay video to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate overlay videos with world state HUD for driving clips"
    )
    parser.add_argument(
        "--clip",
        type=str,
        default="clip1",
        help="Clip name to process (e.g., clip1, clip2, clip3). Default: clip1",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for output video. Default: 2.0",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    preds_path = project_root / "data" / "predictions" / "world_state_claude_smoothed.csv"
    segments_path = project_root / "data" / "predictions" / "world_state_segments.csv"
    gloss_path = project_root / "data" / "predictions" / "segment_gloss.csv"
    planner_path = project_root / "data" / "predictions" / "planner_commands.csv"

    frames_root = project_root / "data" / "frames"
    out_dir = project_root / "results"

    preds = load_smoothed_preds(preds_path)
    segments = load_segments(segments_path)
    gloss = load_gloss(gloss_path)
    planner = load_planner(planner_path)

    make_overlay_video_for_clip(args.clip, frames_root, preds, segments, gloss, planner, out_dir, fps=args.fps)


if __name__ == "__main__":
    main()
