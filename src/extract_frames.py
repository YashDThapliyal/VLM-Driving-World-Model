import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames_from_video(video_path: Path, output_dir: Path, fps: float, overwrite: bool = False) -> None:
    if not video_path.is_file():
        print(f"[WARN] Video not found: {video_path}")
        return

    video_id = video_path.stem  # e.g. "my_drive_01"
    video_output_dir = output_dir / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # If directory already has frames and we're not overwriting, skip
    if not overwrite and any(video_output_dir.iterdir()):
        print(f"[INFO] Skipping {video_id}, frames already exist and --overwrite not set.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print(f"[WARN] Invalid FPS ({video_fps}) for video: {video_path}")
        cap.release()
        return

    frame_interval = max(int(round(video_fps / fps)), 1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[WARN] No frames detected in video: {video_path}")
        cap.release()
        return

    print(f"[INFO] Extracting frames from {video_path.name} at ~{fps} fps into {video_output_dir}")

    frame_idx = 0
    saved_idx = 0

    with tqdm(total=total_frames, desc=f"{video_id}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                saved_idx += 1
                filename = f"frame_{saved_idx:04d}.jpg"
                out_path = video_output_dir / filename
                cv2.imwrite(str(out_path), frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"[INFO] Saved {saved_idx} frames for {video_id}.")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from driving videos.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw_videos",
        help="Directory containing input .mp4 videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/frames",
        help="Directory where extracted frames will be stored.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target frames per second to sample from the video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite/replace existing frames for a given video.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return

    video_files = sorted(p for p in input_dir.glob("*.mp4"))
    if not video_files:
        print(f"[WARN] No .mp4 files found in {input_dir}")
        return

    for vid in video_files:
        extract_frames_from_video(vid, output_dir, fps=args.fps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
