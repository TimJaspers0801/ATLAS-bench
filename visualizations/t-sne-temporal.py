import os
from pathlib import Path
import numpy as np
import torch
import timm
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torchvision import transforms
from models.load_models import load_dinov2_l

# -----------------------------
# Config
# -----------------------------
FRAMES_DIR = Path(r"E:\SurgeSAM_final_full_videos\cholecystectomy\5TEvJOWx7OM_ROBOT_VAL")
IMG_SIZE = 336
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Subsampling
MAX_FRAMES = 5000        # set None to use all frames
SUBSAMPLE_MODE = "uniform"  # "uniform" or "random"


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.43564648, 0.28886865, 0.28611407),
        std=(0.27210076, 0.22871057, 0.22265891)
    )
])

# -----------------------------
# Frame loading
# -----------------------------
def load_frames_sorted(frames_dir):
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")
    return frames

# -----------------------------
# Subsampling
# -----------------------------
def subsample_frames(frames, max_frames=None, mode="uniform"):
    if max_frames is None or len(frames) <= max_frames:
        return frames

    if mode == "uniform":
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        return [frames[i] for i in idx]

    elif mode == "random":
        return list(np.random.choice(frames, max_frames, replace=False))

    else:
        raise ValueError("mode must be 'uniform' or 'random'")

# -----------------------------
# Feature extraction
# -----------------------------
@torch.no_grad()
def extract_features(model, frames):
    feats = []

    for img_path in tqdm(frames):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        feat = model(x)   # (1, 1024)
        feat = feat.squeeze(0).cpu().numpy()
        feats.append(feat)

    return np.stack(feats)

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading frames...")
    frames = load_frames_sorted(FRAMES_DIR)
    print(f"Total frames found: {len(frames)}")

    frames = subsample_frames(frames, MAX_FRAMES, SUBSAMPLE_MODE)
    print(f"Frames after subsampling: {len(frames)}")

    print("Loading model...")
    model = load_dinov2_l().to(DEVICE)

    print("Extracting features...")
    X = extract_features(model, frames)

    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=0,
    )

    X_2d = tsne.fit_transform(X)

    # -----------------------------
    # Temporal color encoding
    # -----------------------------
    # Normalize frame index to [0,1]
    t = np.linspace(0, 1, len(frames))

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(20, 10))

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=t,
        cmap="viridis",   # perceptually uniform, no red/green
        s=18,
        alpha=0.85,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Video position (start → end)", rotation=270, labelpad=20, fontsize=24)

    plt.title("DINOv2 ViT-L t-SNE — Temporal Video Embedding", fontsize=24)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("t-sne_temporal.png", dpi=300)
    plt.savefig("t-sne_temporal.svg")
    plt.savefig("t-sne_temporal.pdf")
    plt.show()


if __name__ == "__main__":
    main()
