import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ------------------------------------------------------------
# Color palette
# ------------------------------------------------------------
color_palette = {
    1: (255, 255, 255), 2: (0, 0, 255), 3: (255, 0, 0), 4: (255, 255, 0), 5: (0, 255, 0),
    6: (0, 200, 100), 7: (200, 150, 100), 8: (250, 150, 100), 9: (255, 200, 100), 10: (180, 0, 0),
    11: (0, 0, 180), 12: (150, 100, 50), 13: (0, 255, 255), 14: (0, 200, 255), 15: (0, 100, 255),
    16: (255, 150, 50), 17: (255, 220, 200), 18: (200, 100, 200), 19: (144, 238, 144), 20: (247, 255, 0),
    21: (255, 206, 27), 22: (200, 0, 200), 23: (255, 0, 150), 24: (255, 100, 200), 25: (200, 100, 255),
    26: (150, 0, 100), 27: (255, 200, 255), 28: (150, 100, 75), 29: (200, 0, 150), 30: (100, 100, 100),
    31: (255, 150, 255), 32: (100, 200, 255), 33: (150, 200, 255), 34: (0, 150, 255), 35: (255, 100, 100),
    36: (200, 200, 255), 37: (100, 100, 255), 38: (0, 255, 150), 39: (255, 255, 100), 40: (150, 150, 150),
    41: (50, 50, 50), 43: (173, 216, 230), 44: (255, 140, 0),
    45: [(252, 186, 3), (223, 3, 252)],
    46: (0, 80, 100),
    0: (0, 0, 0)
}

# ------------------------------------------------------------
# Build fast lookup table: packed RGB -> class id
# ------------------------------------------------------------
rgb2cls = {}

def pack_rgb(rgb):
    r, g, b = rgb
    return (r << 16) | (g << 8) | b

for cls, colors in color_palette.items():
    if isinstance(colors, list):
        for c in colors:
            rgb2cls[pack_rgb(c)] = cls
    else:
        rgb2cls[pack_rgb(colors)] = cls


# ------------------------------------------------------------
# Fast mask conversion
# ------------------------------------------------------------
def convert_mask_fast(mask_path: Path):
    img = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

    # Pack RGB → int
    rgb_packed = (
        (img[..., 0].astype(np.uint32) << 16) |
        (img[..., 1].astype(np.uint32) << 8)  |
        img[..., 2].astype(np.uint32)
    )

    flat = rgb_packed.ravel()
    out = np.zeros_like(flat, dtype=np.uint8)

    unique_colors = np.unique(flat)
    for c in unique_colors:
        if c not in rgb2cls:
            raise ValueError(f"Unknown color {((c>>16)&255, (c>>8)&255, c&255)} in {mask_path}")
        out[flat == c] = rgb2cls[c]

    out = out.reshape(rgb_packed.shape)

    out_dir = mask_path.parent.parent / "machine_masks"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(out).save(out_dir / mask_path.name)


# ------------------------------------------------------------
# Dataset traversal (multiprocessing)
# ------------------------------------------------------------
def process_dataset(root: Path, workers: int = None):
    mask_paths = list(root.glob("**/masks/frame_*.png"))
    print(f"Found {len(mask_paths)} masks")

    workers = workers or max(cpu_count() - 1, 1)

    with Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(convert_mask_fast, mask_paths),
                  total=len(mask_paths)))


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    process_dataset(Path("E:\SurgeSAM_final_split"))
