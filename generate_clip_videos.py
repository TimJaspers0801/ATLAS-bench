"""
Generate an annotated .mp4 video for every clip in the ATLAS dataset zip.

Output layout:
  <OUTPUT_DIR>/<split>/<procedure>/<video>/<clip>.mp4

Each video frame shows the original image (left) next to the mask overlay
(right). The mask PNG is used directly as a color image — no palette lookup.

Usage:
  python generate_clip_videos.py
  python generate_clip_videos.py --zip E:\atlas120k --out E:\atlas120k_annotated_clips --splits train val test --fps 10
"""

import argparse
import os
import sys
import zipfile
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

# ── Defaults ──────────────────────────────────────────────────────────────────
ZIP_PATH = r"E:\atlas120k"
OUTPUT_DIR = r"E:\atlas120k_annotated_clips"
SPLITS = ["train", "val", "test"]
FPS = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_from_zip(zf: zipfile.ZipFile, path: str) -> Image.Image:
    with zf.open(path) as fp:
        data = fp.read()
    return Image.open(BytesIO(data))


def read_image(zf: zipfile.ZipFile, path: str) -> np.ndarray:
    """Return uint8 HxWx3 RGB array."""
    return np.array(_pil_from_zip(zf, path).convert("RGB"))


def read_mask_rgb(zf: zipfile.ZipFile, path: str) -> np.ndarray:
    """Return the mask PNG as a uint8 HxWx3 RGB color array."""
    return np.array(_pil_from_zip(zf, path).convert("RGB"))


def apply_color_mask_overlay(image: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
    """
    Blend a color mask onto an image.
      - Annotated inner area (non-black mask pixels): 50% image + 50% mask
      - Annotation edges: 10% image + 90% mask  (sharp outline)
      - Background (black mask pixels): image unchanged

    image      : uint8 HxWx3 RGB
    color_mask : uint8 HxWx3 RGB (the mask PNG read directly)
    """
    is_annotated = np.any(color_mask > 0, axis=2).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(is_annotated * 255, 50, 150)
    edges = cv2.dilate(edges, kernel, iterations=2).astype(bool)

    blended = image.copy()
    inner = is_annotated.astype(bool) & ~edges
    blended[inner] = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)[inner]
    blended[edges] = cv2.addWeighted(image, 0.1, color_mask, 0.9, 0)[edges]
    return blended


def build_clip_index(zf: zipfile.ZipFile, splits: list[str], root_prefix: str) -> dict:
    """
    Scan the zip and return:
      { clip_id -> [(img_path, mask_path, filename), ...] }
    clip_id format: "{split}/{procedure}/{video}/{clip}"
    Frames within each clip are sorted by filename.
    """
    all_files = {p.lstrip("./") for p in zf.namelist() if not p.endswith("/")}
    clips: dict[str, list] = {}

    for path in all_files:
        low = path.lower()
        if "/images/" not in low or not low.endswith(".jpg"):
            continue

        matched_split = matched_prefix = None
        for split in splits:
            prefix = f"{root_prefix}{split}/"
            if path.startswith(prefix):
                matched_split, matched_prefix = split, prefix
                break
        if matched_split is None:
            continue

        rel = path[len(matched_prefix):]
        parts = rel.split("/")
        if len(parts) < 5:
            continue

        procedure, video, clip, _, filename = parts[0], parts[1], parts[2], parts[3], parts[-1]

        mask_path = (
            f"{matched_prefix}{procedure}/{video}/{clip}/masks/"
            + filename.replace(".jpg", ".png")
        )
        if mask_path not in all_files:
            continue

        clip_id = f"{matched_split}/{procedure}/{video}/{clip}"
        clips.setdefault(clip_id, []).append((path, mask_path, filename))

    for frames in clips.values():
        frames.sort(key=lambda x: x[2])

    return clips


def write_clip_video(
    zf: zipfile.ZipFile,
    frames: list[tuple],
    output_path: str,
    fps: int,
) -> None:
    """Render one clip to an mp4. Each video frame is: original | mask overlay."""
    if not frames:
        return

    img0 = read_image(zf, frames[0][0])
    h, w = img0.shape[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    for img_path, mask_path, _ in frames:
        img = read_image(zf, img_path)
        color_mask = read_mask_rgb(zf, mask_path)
        overlay = apply_color_mask_overlay(img, color_mask)

        # Convert RGB -> BGR for OpenCV
        frame = np.concatenate([img[:, :, ::-1], overlay[:, :, ::-1]], axis=1)
        writer.write(frame)

    writer.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate annotated clip videos from the ATLAS dataset.")
    parser.add_argument("--zip",    default=ZIP_PATH,    help="Path to ATLAS zip file")
    parser.add_argument("--out",    default=OUTPUT_DIR,  help="Output root directory")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=["train", "val", "test"])
    parser.add_argument("--fps",    type=int, default=FPS)
    args = parser.parse_args()

    zip_path = args.zip
    if not os.path.isfile(zip_path):
        if os.path.isfile(zip_path + ".zip"):
            zip_path = zip_path + ".zip"
        else:
            sys.exit(f"[ERROR] Zip file not found: {zip_path}")

    print(f"ATLAS zip : {zip_path}")
    print(f"Output dir: {args.out}")
    print(f"Splits    : {args.splits}")
    print(f"FPS       : {args.fps}")
    print()

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = [p.lstrip("./") for p in zf.namelist() if not p.endswith("/")]
        top_level = {p.split("/")[0] for p in all_names if p.count("/") > 0}
        root_prefix = (list(top_level)[0] + "/") if len(top_level) == 1 else ""

        print(f"Detected root prefix: '{root_prefix}'")
        print("Indexing clips...")
        clips = build_clip_index(zf, args.splits, root_prefix)
        print(f"Found {len(clips)} clips across splits {args.splits}\n")

        try:
            from tqdm import tqdm
            iterator = tqdm(clips.items(), desc="Rendering clips", unit="clip")
        except ImportError:
            iterator = clips.items()

        for i, (clip_id, frames) in enumerate(iterator, 1):
            out_path = os.path.join(args.out, clip_id.replace("/", os.sep) + ".mp4")

            if not hasattr(iterator, "set_postfix"):
                print(f"[{i}/{len(clips)}] {clip_id} ({len(frames)} frames) -> {out_path}")

            write_clip_video(zf, frames, out_path, fps=args.fps)

    print(f"\nDone. Videos saved to: {args.out}")


if __name__ == "__main__":
    main()
