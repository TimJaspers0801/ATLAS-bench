import os
import math
import random

from moviepy import VideoFileClip, CompositeVideoClip
from moviepy.video.fx.Loop import Loop

random.seed(42)
# ==============================
# CONFIG
# ==============================
VIDEO_FOLDER = r"E:\SurgeSAM_final\example_videos"
OUTPUT_VIDEO = r"E:\SurgeSAM_final\videos.mp4"

FINAL_RESOLUTION = (1920, 1080)
SEGMENT_DURATION = 4      # seconds per grid stage
MAX_GRID_SIZE = 25        # e.g. 1,4,9,16,25,...
FPS = 15

GRID_STAGES = [1, 4, 9, 16, 25]   # number of videos shown


# ==============================
# LOAD VIDEOS
# ==============================
video_files = [
    os.path.join(VIDEO_FOLDER, f)
    for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
]

if not video_files:
    raise RuntimeError("No videos found in folder!")

print(f"Found {len(video_files)} videos")

# Load all videos
clips = [VideoFileClip(v) for v in video_files]

# Make them loop infinitely
clips = [clip.with_effects([Loop()]) for clip in clips]


# ==============================
# GRID BUILDER
# ==============================
def build_grid(num_videos, duration):
    """
    Build a grid with num_videos videos (must be a perfect square).
    """
    grid_size = int(math.sqrt(num_videos))
    assert grid_size * grid_size == num_videos, "Grid size must be perfect square"

    cell_w = FINAL_RESOLUTION[0] // grid_size
    cell_h = FINAL_RESOLUTION[1] // grid_size

    grid_clips = []

    for i in range(num_videos):
        clip = random.choice(clips)

        clip = (
            clip.resized((cell_w, cell_h))
                .with_duration(duration)
        )

        row = i // grid_size
        col = i % grid_size

        x = col * cell_w
        y = row * cell_h

        grid_clips.append(clip.with_position((x, y)))

    return CompositeVideoClip(grid_clips, size=FINAL_RESOLUTION)


# ==============================
# BUILD EXPANDING SEQUENCE
# ==============================
timeline = []
current_t = 0

for n in GRID_STAGES:
    print(f"Building grid with {n} videos...")
    segment = build_grid(n, SEGMENT_DURATION)
    timeline.append(segment.with_start(current_t))
    current_t += SEGMENT_DURATION

final_video = CompositeVideoClip(
    timeline,
    size=FINAL_RESOLUTION
).with_duration(current_t).with_fps(FPS)


# ==============================
# EXPORT
# ==============================
final_video.write_videofile(
    OUTPUT_VIDEO,
    codec="libx264",
    audio=False,
    fps=FPS,
    preset="medium",
    threads=8
)

print("Done! Saved as:", OUTPUT_VIDEO)
