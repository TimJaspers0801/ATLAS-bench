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
    'L-hook Electrocautery',
    'Gallbladder'
]

CHOLECSEG8K_MAPPING = {
    0: 0,  # Background
    1: 1,  # Abdominal Wall
    2: 2,  # Liver
    3: 3,  # Gastrointestinal Tract
    4: 4,  # Fat
    5: 5,  # Grasper
    6: 6,  # Connective Tissue (excluded from evaluation)
    7: 5,  # L-hook Electrocautery -> Grasper (consolidated surgical tools)
    8: 8   # Gallbladder
}


class CholecSeg8kDataset(Dataset):
    """
    Zip-based CholecSeg8k dataset with train/val/test split.
    
    Dataset structure in zip:
    - cholecseg8k/
      - Train/
        - frames/
        - masks/
      - Validation/
        - frames/
        - masks/
      - Test/
        - frames/
        - masks/
    
    Frame format: video{XX}_nb_frame_{framenum}_endo.png
    Masks: machine_masks (corresponding .png files)
    
    All frames are treated as part of a single "clip" for compatibility with AtlasDataset interface.
    """

    def __init__(
        self,
        zip_path,
        split,
        transform=None,
        first_frame_only=False,
        frame_percentage=100,
        seed=42,
        normalization_type="none",
    ):
        assert split in {"train", "val", "test"}
        assert normalization_type in {"none", "surgical", "imagenet"}
        
        if not (1 <= frame_percentage <= 100):
            raise ValueError("frame_percentage must be between 1 and 100")

        self.zip_path = zip_path
        self.zf = None
        self.split = split
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

        # Map lowercase split names to actual folder names in zip
        split_folder_map = {
            "train": "Train",
            "val": "Validation",
            "test": "Test"
        }
        self.split_folder = split_folder_map[split]

        # CholecSeg8k normalization stats
        # These are reasonable defaults; can be updated based on actual statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Storage
        self.samples = []
        self.clip_id = f"cholecseg8k/{self.split_folder}"

        # Scan zip and build sample list
        self._build_dataset()

        print(f"[CholecSeg8kDataset]")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Mode:",
              "first frame" if first_frame_only else
              f"{frame_percentage}% frames")

    def _build_dataset(self):
        """Scan zip file and build sample list."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            all_files = [p.lstrip("./") for p in zf.namelist() if not p.endswith("/")]

        frames_dir = f"cholecseg8k/{self.split_folder}/frames/"
        masks_dir = f"cholecseg8k/{self.split_folder}/masks/"

        all_files_set = set(all_files)

        # Collect frame files
        for file in all_files:
            if not file.startswith(frames_dir) or not file.lower().endswith('.png'):
                continue

            filename = file[len(frames_dir):]
            
            # Construct mask path
            mask_file = masks_dir + filename

            if mask_file not in all_files_set:
                continue

            sample = {
                "img": file,
                "mask": mask_file,
                "filename": filename,
                "split": self.split,
            }
            self.samples.append(sample)

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
        mask = self._read_from_zip(self.zf, sample["mask"]).convert("L")

        # Apply class mapping to mask
        mask = np.array(mask)
        mask = self._remap_mask(mask)
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
            "split": sample["split"],
            "filename": sample["filename"],
        }

    def _remap_mask(self, mask):
        """Apply class mapping to mask."""
        remapped = np.zeros_like(mask)
        for src_class, dst_class in CHOLECSEG8K_MAPPING.items():
            remapped[mask == src_class] = dst_class
        return remapped
