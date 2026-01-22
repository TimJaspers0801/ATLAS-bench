import re
from pathlib import Path

ROOT = r"E:\SurgeSAM_final_full_videos"   # <-- change to your dataset root

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


def process_images_folder(images_dir: Path):
    for file in images_dir.iterdir():
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

        print(f"Renaming: {file} → {new_name}")
        file.rename(new_path)


def walk_dataset(root):
    root = Path(root)

    for procedure in root.iterdir():
        if not procedure.is_dir():
            continue

        for clip in procedure.iterdir():
            if not clip.is_dir():
                continue


            process_images_folder(clip)

if __name__ == "__main__":
    walk_dataset(ROOT)
    print("\n✅ Done renaming dataset frames.")
