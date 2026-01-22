import os
import re
from pathlib import Path

ROOT = r"E:\SurgeSAM_final_split"

# Matches: frame_123.jpg, frame_0123.jpg, frame_000123.jpg etc.
FRAME_REGEX = re.compile(r"frame_(\d+)\.(jpg|png)$", re.IGNORECASE)

def pad_frame_name(filename):
    """
    Converts frame_123.jpg → frame_000123.jpg
    """
    match = FRAME_REGEX.match(filename)
    if not match:
        return None

    number = int(match.group(1))
    ext = match.group(2)
    new_name = f"frame_{number:06d}.{ext}"
    return new_name


def process_folder(folder):
    folder = Path(folder)

    for file in folder.iterdir():
        if not file.is_file():
            continue

        new_name = pad_frame_name(file.name)
        if new_name is None:
            continue

        if file.name == new_name:
            continue  # already correct

        new_path = file.with_name(new_name)

        if new_path.exists():
            print(f"⚠️  Skipping (already exists): {new_path}")
            continue

        print(f"Renaming: {file.name} → {new_name}")
        file.rename(new_path)


def walk_dataset(root):
    root = Path(root)

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

                    images_dir = clip / "images"
                    masks_dir  = clip / "masks"

                    if images_dir.exists():
                        process_folder(images_dir)

                    if masks_dir.exists():
                        process_folder(masks_dir)


if __name__ == "__main__":
    walk_dataset(ROOT)
    print("\n✅ Done renaming dataset frames.")
