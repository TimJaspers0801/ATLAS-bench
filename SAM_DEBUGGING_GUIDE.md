# SAM Evaluation Debugging Guide

## Issues Found in eval_sam.py

### 1. **Critical: Frame-by-Frame Processing Loss (SAM2)**
**Problem**: The `process_clip_sam2()` function processes frames individually instead of leveraging SAM2's video memory tracking capability.

**Why it causes Dice=0**:
- Frames 2+ are processed WITHOUT prompts (no clicks)
- SAM2 without any prompt returns empty/zero masks
- This causes all subsequent frame predictions to be empty
- Empty predictions vs non-empty GT = 0 Dice

**Code location**: Lines 234-263 in `process_clip_sam2()`

**What's happening**:
```python
# Frame 1: Has clicks - might work
# Frames 2+: No clicks - SAM doesn't know what to segment
inputs = processor([clip_frames[frame_idx]], return_tensors="pt")
# This returns empty masks because there's no guidance
```

---

### 2. **Silent Failures in Error Handling**
**Problem**: Exceptions are caught with minimal logging, making it hard to diagnose the root cause.

**Result**: Empty masks are returned without telling you why.

---

### 3. **Potential Label Collision**
**Problem**: When multiple classes overlap in predictions, later classes overwrite earlier ones.

```python
for i, class_id in enumerate(class_id_list):
    if i < len(frame_masks):
        mask = frame_masks[i]
        mask_binary = (mask > 0.5).astype(np.uint8)
        combined_mask[mask_binary > 0] = class_id  # Overwrites previous!
```

---

## How to Debug

### Step 1: Run the Debug Script
I've created `debug_sam.py` to help diagnose the issue:

```bash
python debug_sam.py --model sam2-hiera-large --data_path <path_to_atlas.zip> --num_clicks 3
```

This will:
- Process only the first clip (first 3 frames) for speed
- Print detailed information about:
  - Generated clicks
  - SAM model outputs (shapes, value ranges)
  - Prediction masks vs GT masks
  - Per-class Dice/IoU scores for each frame

### Step 2: Run eval_sam.py with Debug Flag
```bash
python eval_sam.py --model sam2-hiera-large --data_path <path_to_atlas.zip> --debug
```

This will print:
- Class clicks being generated
- SAM processor inputs/outputs
- Mask shapes and value distributions
- Unique classes in each prediction

### Step 3: Look for These Warning Signs

1. **"No masks in pred_masks"** - SAM returned no masks
2. **"pixels after threshold=0"** - Mask thresholding eliminated all pixels
3. **"Pred classes={0}"** - Prediction is entirely background/empty
4. **Large discrepancies between GT and Pred classes** - SAM is not segmenting the right objects

---

## Likely Root Causes

### Case 1: Frame-by-Frame Processing (Most Likely)
If you see:
- Frame 0: Has predictions with multiple classes
- Frames 1+: All predictions are 0 (empty)

**This is the issue.** SAM2 is designed for video and should use temporal tracking, not process frames independently.

**Solution**: Use SAM2's built-in video processing capability by passing the entire video clip at once (if supported by the model).

### Case 2: No Clicks Generated
If you see:
- "Class clicks: {}" or "No valid points after filtering!"

SAM has no guidance, so it returns empty masks.

**Solution**: Check that the GT masks have valid class pixels in them.

### Case 3: SAM Not Returning Proper Masks
If processor inputs/outputs look wrong or model returns None, there might be:
- Version mismatch between transformers library and SAM
- Input format incompatibility
- Memory issues

**Solution**: Check SAM documentation for your version and ensure the processor input format is correct.

### Case 4: Threshold Too High
If masks have values 0.1-0.4 but threshold is 0.5, all pixels get filtered.

**Solution**: Lower the threshold or examine actual mask value distributions.

---

## Changes Made to eval_sam.py

1. Added `debug` parameter to `process_clip_sam2()` and `process_clip_sam3()`
2. Added `debug` parameter to `evaluate_sam()`
3. Added `--debug` flag to argument parser
4. Added detailed logging:
   - Input shapes and types
   - Output shapes and value ranges
   - Per-class pixel counts
   - Thresholding results
5. Improved exception handling with traceback printing

---

## Suggested Fixes

### Option A: Fix SAM2 Video Processing (Recommended)
Currently, frames are processed individually. SAM2 should use temporal memory tracking:

```python
# Current (wrong): Process frame-by-frame
for frame_idx in range(len(clip_frames)):
    outputs = model(**inputs)

# Better: Use SAM2's video capabilities
# Check if SAM2 has a video processing mode that maintains state
```

### Option B: Use Consistent Prompting
If SAM2 doesn't support video mode, provide consistent guidance across frames:
- Extract masks from frame 0
- Use those masks as prompts for subsequent frames
- This maintains continuity

### Option C: Try SAM3
SAM3 might have better batch/video processing support:

```bash
python eval_sam.py --model sam3-base --data_path <path> --debug
```

---

## Next Steps

1. **Run debug script** to see what's actually happening
2. **Check the output** - look for the warning signs above
3. **Share the debug output** if you need further help
4. **Consider the suggested fixes** based on what the debug output reveals

The `--debug` flag should quickly reveal which component is failing.
