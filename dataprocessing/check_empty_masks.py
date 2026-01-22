import os
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = r"E:\SurgeSAM_final_split"   # <-- change to your atlas root
OUTPUT_FILE = "empty_masks.txt"


def is_mask_empty(mask_path):
    """
    Returns True if the mask is completely black (all pixels zero).
    Works for RGB, RGBA, etc.
    """
    img = Image.open(mask_path)
    img = img.convert("RGB")   # ensure 3-channel
    arr = np.array(img)

    return np.all(arr == 0)


def find_empty_masks(root):
    root = Path(root)
    empty_masks = []

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue

        for procedure in split_dir.iterdir():
            if not procedure.is_dir():
                continue

            for video in procedure.iterdir():
                if not video.is_dir():
                    continue

                for clip in video.iterdir():
                    if not clip.is_dir():
                        continue

                    masks_dir = clip / "masks"
                    if not masks_dir.exists():
                        continue

                    for mask_file in masks_dir.glob("*.png"):
                        try:
                            if is_mask_empty(mask_file):
                                print(f"EMPTY: {mask_file}")
                                empty_masks.append(str(mask_file))
                        except Exception as e:
                            print(f"⚠️ Failed to read {mask_file}: {e}")

    return empty_masks


if __name__ == "__main__":
    empty = find_empty_masks(ROOT)

    with open(OUTPUT_FILE, "w") as f:
        for path in empty:
            f.write(path + "\n")

    print(f"\n✅ Done. Found {len(empty)} empty masks.")
    print(f"List saved to: {OUTPUT_FILE}")
