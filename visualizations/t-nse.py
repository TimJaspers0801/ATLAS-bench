import os
import random
from pathlib import Path
import numpy as np
import torch
import timm
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from models.load_models import load_dinov2_l
# -----------------------------
# Config
# -----------------------------
ROOT = Path("E:\SurgeSAM_final")
SAMPLES_PER_PROC = 5000
IMG_SIZE = 336
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROCEDURES = [
'cholecystectomy',
'appendectomy',
'esophagectomy',
'rarp',
'colectomy']



# -----------------------------
# Image preprocessing
# -----------------------------
from torchvision import transforms

transform = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.43564648, 0.28886865, 0.28611407),
        std=(0.27210076, 0.22871057, 0.22265891)
    )
])

# -----------------------------
# Dataset traversal
# -----------------------------
def collect_frames(root):
    """
    Returns dict: procedure -> list[image_paths]
    """
    proc_dict = {}

    split_root = root

    for procedure in sorted(os.listdir(split_root)):
        if procedure == "example_videos":
            continue
        if procedure not in PROCEDURES:
            continue
        proc_path = split_root / procedure
        if not proc_path.is_dir():
            continue

        all_frames = []

        for video in proc_path.iterdir():
            for clip in video.iterdir():
                img_dir = clip / "images"
                if not img_dir.exists():
                    continue

                frames = sorted(img_dir.glob("frame_*.jpg"))
                all_frames.extend(frames)

        if len(all_frames) > 0:
            proc_dict[procedure] = all_frames

    return proc_dict

# -----------------------------
# Feature extraction
# -----------------------------
@torch.no_grad()
def extract_features(model, image_paths):
    feats = []

    for img_path in tqdm(image_paths, leave=False):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        feat = model(x)  # (1, 1024) for ViT-L
        feat = feat.squeeze(0).cpu().numpy()

        feats.append(feat)

    return np.stack(feats)

# -----------------------------
# Main
# -----------------------------
def main():
    model = load_dinov2_l().to(DEVICE)

    proc_dict = collect_frames(ROOT)

    all_feats = []
    all_labels = []

    for proc, frames in proc_dict.items():
        print(f"{proc}: {len(frames)} frames found")

        sampled = random.sample(frames, min(SAMPLES_PER_PROC, len(frames)))

        feats = extract_features(model, sampled)

        all_feats.append(feats)
        all_labels.extend([proc] * len(feats))

    X = np.vstack(all_feats)
    y = np.array(all_labels)

    print("Total samples:", X.shape[0])
    print("Embedding dim:", X.shape[1])

    # -----------------------------
    # t-SNE
    # -----------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=0,
    )

    X_2d = tsne.fit_transform(X)

    # -----------------------------
    # Plot
    # -----------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    plt.figure(figsize=(20, 10))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y_enc,
        s=18,
        alpha=0.6,
        cmap="tab20"
    )

    handles, _ = scatter.legend_elements()
    plt.legend(handles, le.classes_, loc="best", fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.title("DINOv2 ViT-L t-SNE — Procedure Level", fontsize=20)
    plt.tight_layout()
    plt.savefig("t-sne_procedure_level.png", dpi=300)
    plt.savefig("t-sne_procedure_level.svg", bbox_inches="tight")
    plt.savefig("t-sne_procedure_level.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
