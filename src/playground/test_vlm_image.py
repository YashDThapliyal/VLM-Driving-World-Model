# src/test_vlm_image.py

#wanto to make sure the vlm is working properly

from pathlib import Path

from vlm_infer import VLMWorldModel


def main():
    # Path to your test image
    script_dir = Path(__file__).parent
    img_path = script_dir / "test_vlm_img.jpeg"

    if not img_path.exists():
        raise FileNotFoundError(f"Test image not found at: {img_path}")

    wm = VLMWorldModel()
    result = wm.infer_frame(img_path)

    print("=== VLM world state for test_vlm_img.png ===")
    print(f"affordance : {result.get('affordance')}")
    print(f"yield_to   : {result.get('yield_to')}")
    print(f"lead_state : {result.get('lead_state')}")
    print()
    print("explanation:")
    print(result.get("explanation", ""))
    print()
    print("raw response:")
    print(result.get("raw", ""))


if __name__ == "__main__":
    main()
