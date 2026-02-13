"""Analyze checkpoint structure to understand compatibility."""

import torch
from pathlib import Path

checkpoint_path = Path("checkpoints/segmentator_vitl.pth")

if not checkpoint_path.exists():
    print(f"Checkpoint not found: {checkpoint_path}")
    exit(1)

print("=" * 80)
print("CHECKPOINT STRUCTURE ANALYSIS")
print("=" * 80)

# Load checkpoint
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"\nCheckpoint type: {type(ckpt)}")
print(f"Keys in checkpoint: {len(list(ckpt.keys()))}")

# Print key structure
print("\nGroup by module:")
modules = {}
for key in ckpt.keys():
    parts = key.split('.')
    module = parts[0]
    if module not in modules:
        modules[module] = []
    modules[module].append(key)

for module in sorted(modules.keys()):
    print(f"\n{module}: ({len(modules[module])} keys)")
    # Show first few keys
    for key in sorted(modules[module])[:3]:
        shape = ckpt[key].shape if isinstance(ckpt[key], torch.Tensor) else type(ckpt[key])
        print(f"  {key}: {shape}")
    if len(modules[module]) > 3:
        print(f"  ... and {len(modules[module]) - 3} more")

# Look for specific important keys
print("\n" + "=" * 80)
print("KEY STRUCTURAL FEATURES")
print("=" * 80)

important_keys = {
    'encoder.backbone.cls_token': 'Class token parameter',
    'encoder.backbone.pos_embed': 'Positional embeddings',
    'encoder.backbone.patch_embed.proj.weight': 'Patch embedding projection',
    'encoder.backbone.norm.weight': 'Layer norm (final)',
    'q.weight': 'Queries (Detectron2 style)',
    'query_embedding.weight': 'Queries (standalone style)',
    'encoder.backbone.reg_token': 'Register tokens (if using)',
}

for key, description in important_keys.items():
    if key in ckpt:
        shape = ckpt[key].shape
        print(f"✓ {description:40s}: {key}")
        print(f"  Shape: {shape}")
    else:
        # Try to find similar keys
        similar = [k for k in ckpt.keys() if key.split('.')[-1] in k]
        if similar:
            print(f"? {description:40s}: NOT FOUND")
            print(f"  Similar keys: {similar[:2]}")
        else:
            print(f"✗ {description:40s}: NOT FOUND")

# Analyze pos_embed
print("\n" + "=" * 80)
print("POS_EMBED ANALYSIS")
print("=" * 80)

if 'encoder.backbone.pos_embed' in ckpt:
    pos_embed = ckpt['encoder.backbone.pos_embed']
    print(f"Shape: {pos_embed.shape}")
    
    # Calculate expected for 256x256 image with 14x14 patches
    num_patches = (256 // 14) ** 2
    print(f"\nFor 256x256 image with 14x14 patches:")
    print(f"  Expected patches: {num_patches}")
    print(f"  Checkpoint has: {pos_embed.shape[1]}")
    print(f"  Has class token: {pos_embed.shape[1] == num_patches + 1}")
    
    print(f"\nFirst token statistics:")
    print(f"  First token mean: {pos_embed[0, 0, :5]}")
    print(f"  First token std: {pos_embed[0, 0].std().item():.6f}")

# Analyze cls_token if it exists
print("\n" + "=" * 80)
print("CLS_TOKEN ANALYSIS")
print("=" * 80)

if 'encoder.backbone.cls_token' in ckpt:
    cls_token = ckpt['encoder.backbone.cls_token']
    print(f"✓ cls_token found in checkpoint")
    print(f"  Shape: {cls_token.shape}")
    print(f"  Values: {cls_token[0, 0, :5]}")
else:
    print(f"✗ cls_token NOT found in checkpoint")
    print(f"  This suggests the checkpoint was trained without class token")

print("\n" + "=" * 80)
