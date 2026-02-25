import random
import zipfile
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors


# CholecSeg8k class mapping
CHOLECSEG8K_CLASS_NAMES = [
    'Black Background',
    'Abdominal Wall',
    'Liver',
    'Gastrointestinal Tract',
    'Fat',
    'Grasper',
    'Connective Tissue',
    'Blood',
    'Cystic Duct',
    'L-hook Electrocautery',
    'Gallbladder',
    'Hepatic Vein',
    'Liver Ligament',
]

CHOLECSEG8K_MAPPING = {
    0: 0,   # Black Background
    1: 1,   # Abdominal Wall
    2: 2,   # Liver
    3: 3,   # Gastrointestinal Tract
    4: 4,   # Fat
    5: 5,   # Grasper
    6: 6,   # Connective Tissue
    7: 7,   # Blood
    8: 8,   # Cystic Duct
    9: 9,   # L-hook Electrocautery
    10: 10, # Gallbladder
    11: 11, # Hepatic Vein
    12: 12, # Liver Ligament
}

COLOR_CLASS_MAPPING = {
    (127, 127, 127): 0,  # Black Background
    (210, 140, 140): 1,  # Abdominal Wall
    (255, 114, 114): 2,  # Liver
    (231, 70, 156): 3,   # Gastrointestinal Tract
    (186, 183, 75): 4,   # Fat
    (170, 255, 0): 5,    # Grasper
    (255, 85, 0): 6,     # Connective Tissue
    (255, 0, 0): 7,      # Blood
    (255, 255, 0): 8,    # Cystic Duct
    (169, 255, 184): 9,  # L-hook Electrocautery
    (255, 160, 165): 10, # Gallbladder
    (0, 50, 128): 11,    # Hepatic Vein
    (111, 74, 0): 12,    # Liver Ligament
}


class CholecSeg8kDataset(Dataset):
    """
    Zip-based CholecSeg8k dataset (no predefined splits).

        Dataset structure in zip (original):
        - video{nb}/
            - video{nb}_{clip_nb}/
                - frame_{framenum}_endo.png
                - frame_{framenum}_endo_mask.png
                - frame_{framenum}_endo_color_mask.png

        Frames and masks live in the same clip folder and are paired by filename.
        Color masks are preferred when present.
    """

    def __init__(
        self,
        zip_path,
        transform=None,
        first_frame_only=False,
        frame_percentage=100,
        seed=42,
        normalization_type="none",
    ):
        assert normalization_type in {"none", "surgical", "imagenet"}
        
        if not (1 <= frame_percentage <= 100):
            raise ValueError("frame_percentage must be between 1 and 100")

        self.zip_path = zip_path
        self.zf = None
        self.transform = transform
        self.first_frame_only = first_frame_only
        self.frame_percentage = frame_percentage
        self.normalization_type = normalization_type
        self._rng = random.Random(seed)

        # Normalization statistics
        if normalization_type == "surgical":
            # Surgical video dataset specific normalization
            self.norm_mean = [0.46888983, 0.29536288, 0.28712815]
            self.norm_std = [0.24689102, 0.21034359, 0.21188641]
        elif normalization_type == "imagenet":
            # ImageNet normalization for pretrained ViT models (DINOv1/v2/v3)
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
        else:
            # No normalization
            self.norm_mean = None
            self.norm_std = None

        # CholecSeg8k normalization stats
        # These are reasonable defaults; can be updated based on actual statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Storage for clip-aware sampling
        self.clip_to_samples = {}  # {clip_id: [sample1, sample2, ...]}
        self.clip_ids = []  # List of all clip IDs
        self.samples = []  # Flattened list of all samples

        # Scan zip and build sample list
        self._build_dataset()

        print(f"[CholecSeg8kDataset]")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Mode:",
              "first frame" if first_frame_only else
              f"{frame_percentage}% frames")

    def _build_dataset(self):
        """Scan zip file and build sample list with clip-aware organization."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            all_files = [p.lstrip("./") for p in zf.namelist() if not p.endswith("/")]

        all_files_set = set(all_files)

        # Collect frame files and organize by video/clip
        for file in all_files:
            if not file.lower().endswith("_endo.png"):
                continue

            # Expect path like: videoXX/videoXX_YY/frame_000000_endo.png
            parts = file.split("/")
            if len(parts) < 3:
                continue

            video_folder = parts[-3]
            clip_folder = parts[-2]
            filename = parts[-1]

            if not video_folder.startswith("video") or not clip_folder.startswith("video"):
                continue

            color_mask_file = file.replace("_endo.png", "_endo_color_mask.png")
            gray_mask_file = file.replace("_endo.png", "_endo_mask.png")

            if color_mask_file in all_files_set:
                mask_file = color_mask_file
                mask_type = "color"
            elif gray_mask_file in all_files_set:
                mask_file = gray_mask_file
                mask_type = "gray"
            else:
                continue

            video_id = video_folder
            clip_nb = clip_folder.split("_")[-1] if "_" in clip_folder else "0"

            clip_id = f"{video_id}/clip_{clip_nb}"

            sample = {
                "img": file,
                "mask": mask_file,
                "mask_type": mask_type,
                "filename": filename,
                "video": video_id,
                "clip": f"clip_{clip_nb}",
                "clip_id": clip_id,
            }

            # Organize by clip
            if clip_id not in self.clip_to_samples:
                self.clip_to_samples[clip_id] = []
                self.clip_ids.append(clip_id)

            self.clip_to_samples[clip_id].append(sample)

        # Flatten samples for iteration
        for clip_id in sorted(self.clip_ids):
            self.samples.extend(self.clip_to_samples[clip_id])

        # Apply frame percentage sampling
        if self.frame_percentage < 100:
            n = max(1, int(len(self.samples) * self.frame_percentage / 100))
            self.samples = self._rng.sample(self.samples, n)

    def __len__(self):
        if self.first_frame_only:
            return 1
        return len(self.samples)

    def _read_from_zip(self, zf, path):
        """Read image from zip file."""
        with zf.open(path) as fp:
            img = Image.open(fp)
            return img.copy()

    def __getitem__(self, idx):
        # Open the zip once per worker and keep it open
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, "r")

        if self.first_frame_only:
            sample = self.samples[0]
        else:
            sample = self.samples[idx]

        # Read image and mask
        image = self._read_from_zip(self.zf, sample["img"]).convert("RGB")

        if sample.get("mask_type") == "color":
            mask_rgb = self._read_from_zip(self.zf, sample["mask"]).convert("RGB")
            mask = self._color_mask_to_index(mask_rgb)
        else:
            mask_gray = self._read_from_zip(self.zf, sample["mask"]).convert("L")
            mask = self._remap_mask(np.array(mask_gray))

        mask = Image.fromarray(mask.astype(np.uint8))

        # Convert to TVTensors
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert to float and normalize
        image = T.functional.to_dtype(image, dtype=torch.float32, scale=True)
        
        # Apply normalization if specified
        if self.norm_mean is not None and self.norm_std is not None:
            image = T.functional.normalize(image, mean=self.norm_mean, std=self.norm_std)

        return {
            "image": image,
            "mask": mask,
            "filename": sample["filename"],
            "procedure": "cholecseg8k",  # For compatibility with ATLAS evaluation
            "video": sample["video"],  # video01, video02, etc.
            "clip": sample["clip"],  # clip_0, clip_1, etc.
        }

    def _color_mask_to_index(self, mask_rgb):
        """Convert RGB color mask to class indices."""
        mask_arr = np.array(mask_rgb)
        remapped = np.zeros((mask_arr.shape[0], mask_arr.shape[1]), dtype=np.uint8)

        for color, class_id in COLOR_CLASS_MAPPING.items():
            color_arr = np.array(color, dtype=np.uint8)
            matches = np.all(mask_arr == color_arr, axis=-1)
            remapped[matches] = class_id

        return remapped

    def _remap_mask(self, mask):
        """Apply class mapping to mask."""
        remapped = np.zeros_like(mask)
        for src_class, dst_class in CHOLECSEG8K_MAPPING.items():
            remapped[mask == src_class] = dst_class
        return remapped
