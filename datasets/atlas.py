import random
import zipfile
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T


class AtlasDataset(Dataset):
    """
    Zip-based dataset with explicit train/val/test split,
    clip-aware sampling, and rich metadata outputs.
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
    ):
        assert split in {"train", "val", "test"}

        if not (1 <= frame_percentage <= 100):
            raise ValueError("frame_percentage must be between 1 and 100")

        self.zip_path = zip_path
        self.split = split
        self.transform = transform
        self.allowed_videos = allowed_videos
        self.one_sample_per_clip = one_sample_per_clip
        self.first_frame_only = first_frame_only
        self.frame_percentage = frame_percentage

        self._rng = random.Random(seed)

        # Normalize root prefix
        if root_in_zip:
            self.root_prefix = root_in_zip.strip("/\\") + "/"
        else:
            self.root_prefix = ""

        # Prefix INCLUDING split
        self.split_prefix = f"{self.root_prefix}{split}/"

        # Storage
        self.clip_to_samples = {}
        self.clip_ids = []
        self.samples = []

        # Dataset normalization
        self.mean = [0.46888983, 0.29536288, 0.28712815]
        self.std = [0.24689102, 0.21034359, 0.21188641]

        # --------------------------------------------------
        # Scan zip once
        # --------------------------------------------------
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            all_files = [p.lstrip("./") for p in zf.namelist() if not p.endswith("/")]

        all_files_set = set(all_files)

        for file in all_files:
            if not file.startswith(self.split_prefix):
                continue

            low = file.lower()
            if "/images/" not in low or not low.endswith(".jpg"):
                continue

            rel = file[len(self.split_prefix):]
            parts = rel.split("/")

            # procedure/video/clip/images/frame.jpg
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

        # Summary
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
        if self.first_frame_only:
            sample = self.clip_to_samples[self.clip_ids[idx]][0]
        elif self.one_sample_per_clip:
            sample = self._rng.choice(self.clip_to_samples[self.clip_ids[idx]])
        else:
            sample = self.samples[idx]

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            image = self._read_from_zip(zf, sample["img"]).convert("RGB")
            mask = self._read_from_zip(zf, sample["mask"]).convert("L")

        # Convert to tensors
        image = T.ToImage()(image)
        mask = torch.from_numpy(np.array(mask)).long()

        if self.transform:
            image, mask = self.transform(image, mask)

        image = T.Normalize(self.mean, self.std)(image)

        return {
            "image": image,
            "mask": mask,
            "procedure": sample["procedure"],
            "video": sample["video"],
            "clip": sample["clip"],
            "filename": sample["filename"],
        }

