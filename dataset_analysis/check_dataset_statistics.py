import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
import tqdm
import multiprocessing
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import cv2
from itertools import chain
from more_itertools import chunked

# ===========================
# Plot styling
# ===========================
def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'figure.figsize': (12, 8),
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.dpi': 100,
        'axes.edgecolor': 'gray',
        'axes.linewidth': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'axes.grid': False,
        'axes.facecolor': 'white',
    })

# ===========================
# Color palette & class names
# ===========================
color_palette = {
    1: (255, 255, 255), 2: (0, 0, 255), 3: (255, 0, 0), 4: (255, 255, 0),
    5: (0, 255, 0), 6: (0, 200, 100), 7: (200, 150, 100), 8: (250, 150, 100),
    9: (255, 200, 100), 10: (180, 0, 0), 11: (0, 0, 180), 12: (150, 100, 50),
    13: (0, 255, 255), 14: (0, 200, 255), 15: (0, 100, 255), 16: (255, 150, 50),
    17: (255, 220, 200), 18: (200, 100, 200), 19: (144, 238, 144),
    20: (247, 255, 0), 21: (255, 206, 27), 22: (200, 0, 200),
    23: (255, 0, 150), 24: (255, 100, 200), 25: (200, 100, 255),
    26: (150, 0, 100), 27: (255, 200, 255), 28: (150, 100, 75),
    29: (200, 0, 150), 30: (100, 100, 100), 31: (255, 150, 255),
    32: (100, 200, 255), 33: (150, 200, 255), 34: (0, 150, 255),
    35: (255, 100, 100), 36: (200, 200, 255), 37: (100, 100, 255),
    38: (0, 255, 150), 39: (255, 255, 100), 41: (50, 50, 50),
    43: (173, 216, 230), 44: (255, 140, 0), 45: (252, 186, 3),
}

class_names = {
    1: "Tools/camera", 2: "Vein (major)", 3: "Artery (major)", 4: "Nerve (major)",
    5: "Small intestine", 6: "Colon/rectum", 7: "Abdominal wall", 8: "Diaphragm",
    9: "Omentum", 10: "Aorta", 11: "Vena cava", 12: "Liver", 13: "Cystic duct",
    14: "Gallbladder", 15: "Hepatic vein", 16: "Hepatic ligament", 17: "Cystic plate",
    18: "Stomach", 19: "Ductus choledochus", 20: "Mesenterium",
    21: "Ductus hepaticus", 22: "Spleen", 23: "Uterus", 24: "Ovary",
    25: "Oviduct", 26: "Prostate", 27: "Urethra", 28: "Ligated plexus",
    29: "Seminal vesicles", 30: "Catheter", 31: "Bladder", 32: "Kidney",
    33: "Lung", 34: "Airway (bronchus/trachea)", 35: "Esophagus",
    36: "Pericardium", 37: "V azygos", 38: "Thoracic duct", 39: "Nerves",
    41: "Non anatomical structures", 43: "Mesocolon",
    44: "Adrenal Gland", 45: "Pancreas",
}

# ===========================
# Helpers to gather dataset structure
# ===========================
def gather_masks_and_metadata(root_path: Path, skip_folders=None):
    """
    Walks root (expected layout: root/<split>/<procedure>/<video>/<clip>/masks/frame_*.png)
    Returns:
      - all_mask_files: list of Path objects
      - metadata_map: dict str(path) -> (split, procedure, video, clip)
      - structure_stats: dict per-split stats (procedures, videos, clips, annotated_frames, clips_per_procedure, frames_per_procedure)
    """
    if skip_folders is None:
        skip_folders = {"example_videos"}

    all_mask_files = []
    metadata_map = {}
    structure_stats = {}  # split -> stats dict

    for split_dir in sorted(root_path.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        structure_stats[split_name] = {
            "total_procedures": 0,
            "total_videos": 0,
            "total_clips": 0,
            "total_annotated_frames": 0,
            "clips_per_procedure": defaultdict(int),
            "frames_per_procedure": defaultdict(int),
        }

        # For each procedure inside the split
        for procedure in split_dir.iterdir():
            if not procedure.is_dir() or procedure.name in skip_folders:
                continue
            structure_stats[split_name]["total_procedures"] += 1

            videos_in_proc = [p for p in procedure.iterdir() if p.is_dir()]
            structure_stats[split_name]["total_videos"] += len(videos_in_proc)

            for video in videos_in_proc:
                if not video.is_dir():
                    continue
                for clip in video.iterdir():
                    if not clip.is_dir():
                        continue
                    images_folder = clip / "images"
                    masks_folder = clip / "masks"
                    # A clip counts if it has an images folder with files OR masks folder
                    has_images = images_folder.is_dir() and any(images_folder.iterdir())
                    has_masks = masks_folder.is_dir() and any(masks_folder.iterdir())
                    if not (has_images or has_masks):
                        continue

                    structure_stats[split_name]["total_clips"] += 1
                    # count masks if present
                    if masks_folder.is_dir():
                        mask_files = sorted(masks_folder.glob("frame_*.png"))
                        n_masks = len(mask_files)
                        structure_stats[split_name]["total_annotated_frames"] += n_masks
                        structure_stats[split_name]["clips_per_procedure"][procedure.name] += 1
                        structure_stats[split_name]["frames_per_procedure"][procedure.name] += n_masks
                        for mask_path in mask_files:
                            all_mask_files.append(mask_path)
                            metadata_map[str(mask_path)] = (split_name, procedure.name, video.name, clip.name)

    return all_mask_files, metadata_map, structure_stats

# ===========================
# Mask processing (detect classes present in each mask)
# ===========================
def process_mask_file(mask_file_path_str):
    """Standalone function for multiprocessing.Pool. Returns (str_path, set_of_class_ids)."""
    try:
        # read with cv2
        arr = cv2.imread(mask_file_path_str, cv2.IMREAD_UNCHANGED)
        if arr is None:
            # could not read file
            return mask_file_path_str, set()
        # if mask has alpha or single-channel, handle appropriately
        if arr.ndim == 2:  # single channel: maybe class ids - attempt to map unique values not colors
            unique_vals = np.unique(arr)
            # If values are small (<256) and correspond to indexed classes, we can treat them as class ids:
            class_ids = set(int(v) for v in unique_vals if v != 0)
            return mask_file_path_str, class_ids

        # convert BGR -> RGB
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        classes_in_image = set()
        # check for each palette color
        for class_id, color in color_palette.items():
            # vectorized check
            if np.any(np.all(arr_rgb == color, axis=-1)):
                classes_in_image.add(class_id)
        return mask_file_path_str, classes_in_image
    except Exception as e:
        # return empty if error
        return mask_file_path_str, set()

def count_class_occurrences_from_lists(all_mask_files, metadata_map, max_workers=None):
    """
    Count class occurrences at frame, clip and video level using the provided mask list + metadata map.
    Returns: (class_image_counts, class_clip_counts, class_video_counts)
    """
    class_image_counts = defaultdict(int)
    class_clip_counts = defaultdict(int)
    class_video_counts = defaultdict(int)

    # multiprocessing pool
    n_procs = max_workers or max(1, multiprocessing.cpu_count() - 1)

    # Convert Paths to strings for pool
    all_mask_strs = [str(p) for p in all_mask_files]

    results = []
    with multiprocessing.Pool(processes=n_procs) as pool:
        for res in tqdm.tqdm(pool.imap(process_mask_file, all_mask_strs), total=len(all_mask_strs), desc="Processing masks"):
            results.append(res)

    # collect per-clip and per-video sets
    clip_class_map = defaultdict(set)
    video_class_map = defaultdict(set)

    for mask_path_str, classes_in_image in results:
        # increment per-frame counts
        for class_id in classes_in_image:
            class_image_counts[class_id] += 1

        # metadata contains (split, procedure, video, clip) - we only need video and clip id for grouping
        meta = metadata_map.get(mask_path_str)
        if meta is None:
            continue
        split_name, procedure_name, video_name, clip_name = meta
        clip_key = (split_name, procedure_name, video_name, clip_name)
        video_key = (split_name, procedure_name, video_name)

        for class_id in classes_in_image:
            clip_class_map[clip_key].add(class_id)
            video_class_map[video_key].add(class_id)

    # sum clip-level occurrences
    for class_ids in clip_class_map.values():
        for class_id in class_ids:
            class_clip_counts[class_id] += 1

    # sum video-level occurrences
    for class_ids in video_class_map.values():
        for class_id in class_ids:
            class_video_counts[class_id] += 1

    return class_image_counts, class_clip_counts, class_video_counts

# ===========================
# Utilities to transform counts to DataFrames
# ===========================
def class_counts_to_df(class_image_counts, class_clip_counts, class_video_counts):
    class_ids = sorted(set(chain(class_image_counts.keys(), class_clip_counts.keys(), class_video_counts.keys())))
    rows = []
    for cid in class_ids:
        rows.append({
            "Class ID": cid,
            "Class Name": class_names.get(cid, "Unknown"),
            "Frame Count": int(class_image_counts.get(cid, 0)),
            "Clip Count": int(class_clip_counts.get(cid, 0)),
            "Video Count": int(class_video_counts.get(cid, 0)),
        })
    return pd.DataFrame(rows)

def structure_stats_to_df(structure_stats):
    # structure_stats is the dict for a split
    df = pd.DataFrame({
        "Statistic": [
            "Total procedures",
            "Total videos",
            "Total clips",
            "Total annotated frames"
        ],
        "Value": [
            structure_stats["total_procedures"],
            structure_stats["total_videos"],
            structure_stats["total_clips"],
            structure_stats["total_annotated_frames"]
        ]
    })
    return df

# ===========================
# Plotting
# ===========================
def plot_class_occurrences(class_image_counts, class_clip_counts, class_video_counts, title_prefix="Overall"):
    set_plot_style()
    data = {
        "Frame Count": class_image_counts,
        "Clip Count": class_clip_counts,
        "Video Count": class_video_counts
    }

    # produce one plot per metric
    for title, counts in data.items():
        if not counts:
            continue
        plt.figure(figsize=(12, 6))
        class_ids = sorted(counts.keys())
        values = [counts[class_id] for class_id in class_ids]
        labels = [class_names.get(class_id, f"Class {class_id}") for class_id in class_ids]

        bars = plt.barh(labels, values)
        plt.xlabel(title)
        plt.ylabel("Anatomical Structure")
        plt.title(f"{title_prefix} - Class Occurrences by {title}")

        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     str(int(bar.get_width())), va='center', ha='left', fontsize=10)

        plt.tight_layout()
        plt.show()

# ===========================
# Main entry: compute overall + per-split stats and save Excel
# ===========================
def analyze_dataset(root_folder: str, output_excel: str = "dataset_statistics_splits.xlsx", max_workers=None, plot=False):
    root_path = Path(root_folder)
    if not root_path.exists():
        raise FileNotFoundError(f"Root folder not found: {root_folder}")

    print("Gathering masks and structure info...")
    all_mask_files, metadata_map, per_split_structure = gather_masks_and_metadata(root_path)

    # Aggregate overall structure stats
    overall_structure = {
        "total_procedures": 0,
        "total_videos": 0,
        "total_clips": 0,
        "total_annotated_frames": 0,
        "clips_per_procedure": defaultdict(int),
        "frames_per_procedure": defaultdict(int),
    }
    for split_name, stats in per_split_structure.items():
        overall_structure["total_procedures"] += stats["total_procedures"]
        overall_structure["total_videos"] += stats["total_videos"]
        overall_structure["total_clips"] += stats["total_clips"]
        overall_structure["total_annotated_frames"] += stats["total_annotated_frames"]
        # merge per procedure counts (note: procedure names may collide across splits; keep them separate if needed)
        for proc, c in stats["clips_per_procedure"].items():
            overall_structure["clips_per_procedure"][f"{split_name}/{proc}"] += c
        for proc, f in stats["frames_per_procedure"].items():
            overall_structure["frames_per_procedure"][f"{split_name}/{proc}"] += f

    # Count class occurrences overall (across all splits)
    print("Counting class occurrences (overall)...")
    class_image_counts_overall, class_clip_counts_overall, class_video_counts_overall = \
        count_class_occurrences_from_lists(all_mask_files, metadata_map, max_workers=max_workers)

    # Count class occurrences per split by filtering mask lists
    per_split_class_counts = {}
    for split_name in per_split_structure.keys():
        print(f"Counting class occurrences for split: {split_name} ...")
        # filter masks for this split
        split_mask_files = [p for p in all_mask_files if metadata_map[str(p)][0] == split_name]
        # build split metadata map
        split_meta = {str(p): metadata_map[str(p)] for p in split_mask_files}
        ci, cc, cv = count_class_occurrences_from_lists(split_mask_files, split_meta, max_workers=max_workers)
        per_split_class_counts[split_name] = (ci, cc, cv)

    # Save everything to Excel with multiple sheets
    print(f"Writing results to Excel: {output_excel}")
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        # Overall structure
        overall_struct_df = structure_stats_to_df(overall_structure)
        overall_struct_df.to_excel(writer, index=False, sheet_name="Overall_Structure")

        # Per-split structure sheets
        for split_name, stats in per_split_structure.items():
            df_stats = structure_stats_to_df(stats)
            df_stats.to_excel(writer, index=False, sheet_name=f"{split_name}_Structure")

        # Overall class stats
        df_overall_classes = class_counts_to_df(class_image_counts_overall, class_clip_counts_overall, class_video_counts_overall)
        df_overall_classes.to_excel(writer, index=False, sheet_name="Overall_Class_Stats")

        # Per-split class stats
        for split_name, (ci, cc, cv) in per_split_class_counts.items():
            df_split = class_counts_to_df(ci, cc, cv)
            # sheet names must be <=31 chars
            sheet_name = f"{split_name[:20]}_Class_Stats"
            df_split.to_excel(writer, index=False, sheet_name=sheet_name)

    print("\nSummary:")
    print(f"Overall procedures: {overall_structure['total_procedures']}")
    print(f"Overall videos: {overall_structure['total_videos']}")
    print(f"Overall clips: {overall_structure['total_clips']}")
    print(f"Overall annotated frames: {overall_structure['total_annotated_frames']}")
    for split_name, stats in per_split_structure.items():
        print(f"\nSplit '{split_name}':")
        print(f"  procedures: {stats['total_procedures']}")
        print(f"  videos: {stats['total_videos']}")
        print(f"  clips: {stats['total_clips']}")
        print(f"  annotated frames: {stats['total_annotated_frames']}")

    if plot:
        # plots for overall and each split
        plot_class_occurrences(class_image_counts_overall, class_clip_counts_overall, class_video_counts_overall, title_prefix="Overall")
        for split_name, (ci, cc, cv) in per_split_class_counts.items():
            plot_class_occurrences(ci, cc, cv, title_prefix=split_name)

    print(f"\n✅ Excel saved to: {os.path.abspath(output_excel)}")
    return {
        "overall_structure": overall_structure,
        "per_split_structure": per_split_structure,
        "overall_class_counts": (class_image_counts_overall, class_clip_counts_overall, class_video_counts_overall),
        "per_split_class_counts": per_split_class_counts,
        "excel_path": os.path.abspath(output_excel)
    }

# ===========================
# Fast scan for videos, clips and frames
# ===========================
def fast_video_clip_frame_scan(root_path: Path, skip_folders=None):
    if skip_folders is None:
        skip_folders = {"example_videos"}

    per_split_counts = {}
    total_videos = 0
    total_clips = 0
    total_frames = 0

    for split_dir in sorted(root_path.iterdir()):
        if not split_dir.is_dir():
            continue

        split_name = split_dir.name
        videos = 0
        clips = 0
        frames = 0

        for procedure in split_dir.iterdir():
            if not procedure.is_dir() or procedure.name in skip_folders:
                continue

            for video in procedure.iterdir():
                if not video.is_dir():
                    continue
                videos += 1

                for clip in video.iterdir():
                    if not clip.is_dir():
                        continue

                    images_folder = clip / "images"
                    masks_folder = clip / "masks"

                    has_images = images_folder.is_dir() and any(images_folder.iterdir())
                    has_masks = masks_folder.is_dir() and any(masks_folder.iterdir())

                    if not (has_images or has_masks):
                        continue

                    clips += 1

                    if has_masks:
                        frames += len(list(masks_folder.glob("frame_*.png")))

        per_split_counts[split_name] = {
            "videos": videos,
            "clips": clips,
            "frames": frames
        }

        total_videos += videos
        total_clips += clips
        total_frames += frames

    return total_videos, total_clips, total_frames, per_split_counts


if __name__ == "__main__":
    # CHANGE this to your dataset root
    ROOT_FOLDER = r"E:\SurgeSam_final_split"   # e.g. root folder that contains train, val, test
    OUTPUT_EXCEL = os.path.join(os.getcwd(), "dataset_statistics_splits.xlsx")
    MAX_WORKERS = None

    root_path = Path(ROOT_FOLDER)

    # -------------------------------------------------
    # FAST SCAN FIRST (videos + clips + frames)
    # -------------------------------------------------
    print("\n===== DATASET OVERVIEW =====")
    total_videos, total_clips, total_frames, split_counts = fast_video_clip_frame_scan(root_path)

    print(f"Overall videos: {total_videos}")
    print(f"Overall clips:  {total_clips}")
    print(f"Overall frames: {total_frames}")

    for split, counts in split_counts.items():
        print(f"\nSplit: {split}")
        print(f"  Videos: {counts['videos']}")
        print(f"  Clips:  {counts['clips']}")
        print(f"  Frames: {counts['frames']}")

    print("===========================\n")

    # -------------------------------------------------
    # FULL ANALYSIS (class stats, Excel, etc.)
    # -------------------------------------------------
    results = analyze_dataset(
        ROOT_FOLDER,
        output_excel=OUTPUT_EXCEL,
        max_workers=MAX_WORKERS,
        plot=True
    )
