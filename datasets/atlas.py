import random
import zipfile
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from .class_mapping import remap_mask, mapping


class AtlasDataset(Dataset):
    """
    Zip-based dataset with explicit train/val/test split,
    clip-aware sampling, and rich metadata outputs.
    Auto-detects top-level folder if root_in_zip is not provided.
    """

    def __init__(
        self,
        zip_path,
        split,
        root_in_zip=None,
        transform=None,
        allowed_videos=None,
        one_sample_per_clip=False,
        first_frame_only=False,
        frame_percentage=100,
        seed=42,
        normalization_type="surgical",
    ):
        assert split in {"train", "val", "test"}
        assert normalization_type in {"none", "surgical", "imagenet"}

        if not (1 <= frame_percentage <= 100):
            raise ValueError("frame_percentage must be between 1 and 100")

        self.zip_path = zip_path
        self.zf = None
        self.split = split
        self.transform = transform
        self.allowed_videos = allowed_videos
        self.one_sample_per_clip = one_sample_per_clip
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

        # Keep original dataset stats for backward compatibility
        self.mean = [0.46888983, 0.29536288, 0.28712815]
        self.std = [0.24689102, 0.21034359, 0.21188641]
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            all_files = [p.lstrip("./") for p in zf.namelist() if not p.endswith("/")]
            top_level = {p.split("/")[0] for p in all_files if p.count("/") > 0}

        if root_in_zip:
            self.root_prefix = root_in_zip.strip("/\\") + "/"
        else:
            # Use single top-level folder if exactly one exists
            if len(top_level) == 1:
                self.root_prefix = list(top_level)[0] + "/"
            else:
                self.root_prefix = ""

        # Prefix including split
        self.split_prefix = f"{self.root_prefix}{split}/"

        # Storage
        self.clip_to_samples = {}
        self.clip_ids = []
        self.samples = []

        # Dataset normalization
        self.mean = [0.46888983, 0.29536288, 0.28712815]
        self.std = [0.24689102, 0.21034359, 0.21188641]

        # --------------------------------------------------
        # Scan zip and build mapping
        # --------------------------------------------------
        all_files_set = set(all_files)
        for file in all_files:
            if not file.startswith(self.split_prefix):
                continue

            low = file.lower()
            if "/images/" not in low or not low.endswith(".jpg"):
                continue

            rel = file[len(self.split_prefix):]
            parts = rel.split("/")

            if len(parts) < 5:
                continue

            procedure, video, clip = parts[0], parts[1], parts[2]
            video_id = f"{procedure}/{video}"

            if self.allowed_videos and video_id not in self.allowed_videos:
                continue

            filename = parts[-1]
            mask_rel = "/".join(
                [procedure, video, clip, "machine_masks", filename.replace(".jpg", ".png")]
            )
            mask_full = self.split_prefix + mask_rel

            if mask_full not in all_files_set:
                continue

            clip_id = f"{procedure}/{video}/{clip}"

            sample = {
                "img": file,
                "mask": mask_full,
                "procedure": procedure,
                "video": video,
                "clip": clip,
                "filename": filename,
            }

            self.clip_to_samples.setdefault(clip_id, []).append(sample)

        # --------------------------------------------------
        # Build index
        # --------------------------------------------------
        self.clip_ids = sorted(self.clip_to_samples.keys())

        if not (self.first_frame_only or self.one_sample_per_clip):
            for clip_id in self.clip_ids:
                samples = self.clip_to_samples[clip_id]
                if frame_percentage < 100:
                    n = max(1, int(len(samples) * frame_percentage / 100))
                    samples = self._rng.sample(samples, n)
                self.samples.extend(samples)

        videos = {cid.rsplit("/", 1)[0] for cid in self.clip_ids}

        print(f"[AtlasDataset]")
        print(f"  Split: {split}")
        print(f"  Clips: {len(self.clip_ids)}")
        print(f"  Videos: {len(videos)}")
        print(f"  Mode:",
              "first frame" if first_frame_only else
              "one per clip" if one_sample_per_clip else
              f"{frame_percentage}% frames")


    def __len__(self):
        if self.first_frame_only or self.one_sample_per_clip:
            return len(self.clip_ids)
        return len(self.samples)

    def _read_from_zip(self, zf, path):
        with zf.open(path) as fp:
            img = Image.open(fp)
            return img.copy()

    def __getitem__(self, idx):
        # Open the zip once per worker and keep it open
        if self.zf is None:
            self.zf = zipfile.ZipFile(self.zip_path, "r")

        if self.first_frame_only:
            sample = self.clip_to_samples[self.clip_ids[idx]][0]
        elif self.one_sample_per_clip:
            sample = self._rng.choice(self.clip_to_samples[self.clip_ids[idx]])
        else:
            sample = self.samples[idx]

        image = self._read_from_zip(self.zf, sample["img"]).convert("RGB")
        mask = self._read_from_zip(self.zf, sample["mask"]).convert("L")

        # Apply class mapping to mask
        mask = np.array(mask)
        mask = remap_mask(mask, mapping)
        mask = Image.fromarray(mask.astype(np.uint8))

        # 1. Convert to TVTensors instead of using functional.to_mask
        # This automatically converts PIL -> Tensor and tags the type
        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        # 2. Apply transforms (v2 handles (Tensor, Tensor) pairs perfectly)
        if self.transform:
            image, mask = self.transform(image, mask)

        # 3. Convert to float and SCALE to [0, 1]
        # The 'scale=True' is the critical missing piece!
        image = T.functional.to_dtype(image, dtype=torch.float32, scale=True)

        # 4. Apply normalization if specified
        if self.norm_mean is not None and self.norm_std is not None:
            image = T.functional.normalize(image, mean=self.norm_mean, std=self.norm_std)

        return {
            "image": image,
            "mask": mask,
            "procedure": sample["procedure"],
            "video": sample["video"],
            "clip": sample["clip"],
            "filename": sample["filename"],
        }

