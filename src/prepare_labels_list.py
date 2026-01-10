import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", default="data/frames", type=str)
    parser.add_argument(
        "--video_id",
        required=True,
        type=str,
        help="Subfolder under frames_root, e.g. 'clip1'",
    )
    parser.add_argument(
        "--step",
        default=1,
        type=int,
        help="Take every Nth frame (default: 1).",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_root) / args.video_id
    if not frames_dir.is_dir():
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        return

    # Frames should be named like frame_XXXX.jpg
    frames = sorted(p.name for p in frames_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"[ERROR] No frames found in {frames_dir}")
        return

    frames = frames[:: args.step]

    # New world model schema
    print("frame_id,affordance,yield_to,lead_state")
    for fname in frames:
        frame_id = f"{args.video_id}/{fname}"
        # placeholders
        print(f"{frame_id},unknown,unknown,unknown")

if __name__ == "__main__":
    main()
