"""Debug script to analyze token ordering in VideoMT."""

import torch
from models.videomt.videomt_standalone import VideoMT

# Create model with pretrained ViT
model = VideoMT(
    img_size=256,
    num_classes=30,
    num_queries=100,
    model_name='vit_large_patch14_dinov2.lvd142m',
    segmenter_blocks=[22, 23],
    task='vss',
    device='cpu'
)

# Check the backbone structure
backbone = model.encoder.backbone

print("=" * 80)
print("BACKBONE CONFIGURATION")
print("=" * 80)
print(f"num_prefix_tokens: {backbone.num_prefix_tokens}")
print(f"Has cls_token: {hasattr(backbone, 'cls_token')}")
if hasattr(backbone, 'cls_token'):
    print(f"cls_token shape: {backbone.cls_token.shape}")

# Create a test frame
test_frame = torch.randn(1, 3, 256, 256)

print("\n" + "=" * 80)
print("TOKEN SEQUENCES DURING FORWARD PASS")
print("=" * 80)

# Step through forward pass manually
x = backbone.patch_embed(test_frame)
_, H, W, _ = x.shape
print(f"\nAfter patch_embed: shape = {x.shape}")
print(f"  - This should be (B, num_spatial_tokens, embed_dim)")
print(f"  - num_spatial_tokens = H*W = {H*W}")

x = backbone._pos_embed(x)
print(f"After pos_embed: shape = {x.shape}")

x = backbone.patch_drop(x)
print(f"After patch_drop: shape = {x.shape}")

x = backbone.norm_pre(x)
print(f"After norm_pre: shape = {x.shape}")

# Process initial backbone blocks
start_block = model.segmenter_blocks[0]
print(f"\nProcessing backbone blocks 0 to {start_block}")
for i, block in enumerate(backbone.blocks[0:start_block]):
    x = x + block.drop_path1(
        block.ls1(backbone.attn(block.norm1(x)))
    )
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

print(f"After backbone blocks: shape = {x.shape}")

# THIS IS THE CRITICAL POINT
print("\n" + "=" * 80)
print("QUERY CONCATENATION (CRITICAL POINT)")
print("=" * 80)

queries = model.query_embedding.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
print(f"Queries shape: {queries.shape}")
print(f"  - num_queries = {model.num_queries}")

print(f"\nSpatial tokens (before concatenation): shape = {x.shape}")
x_concatenated = torch.cat((queries, x), dim=1)
print(f"After concatenation: shape = {x_concatenated.shape}")
print(f"  - Expected: (B, num_queries + num_spatial, embed_dim)")
print(f"  - Expected: (1, {model.num_queries} + {H*W}, {backbone.embed_dim})")

# Process through segmenter blocks
x = x_concatenated
segmenter_start = model.segmenter_blocks[0]
segmenter_end = model.segmenter_blocks[-1] + 1
print(f"\nProcessing segmenter blocks {segmenter_start} to {segmenter_end}")
for i, block in enumerate(backbone.blocks[segmenter_start:segmenter_end]):
    x = x + block.drop_path1(
        block.ls1(backbone.attn(block.norm1(x)))
    )
    x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

x = backbone.norm(x)
print(f"After segmenter blocks and norm: shape = {x.shape}")

print("\n" + "=" * 80)
print("SPATIAL FEATURE EXTRACTION")
print("=" * 80)

q = x[:, :model.num_queries, :]
print(f"Query features (first {model.num_queries} tokens): shape = {q.shape}")

num_prefix = backbone.num_prefix_tokens
print(f"num_prefix_tokens = {num_prefix}")

spatial_tokens = x[:, model.num_queries + num_prefix:, :]
print(f"Spatial tokens (skip queries + prefix): shape = {spatial_tokens.shape}")
print(f"  - Expected: (B, H*W, embed_dim) = (1, {H*W}, {backbone.embed_dim})")

if spatial_tokens.shape[1] != H*W:
    print(f"\n⚠️  WARNING: Spatial tokens shape mismatch!")
    print(f"  - Expected {H*W} spatial tokens")
    print(f"  - Got {spatial_tokens.shape[1]} spatial tokens")
    print(f"\n  Token breakdown AFTER concatenation:")
    print(f"    [0:{model.num_queries}] = query tokens ({model.num_queries} tokens)")
    print(f"    [{model.num_queries}:{model.num_queries + num_prefix}] = class token ({num_prefix} tokens)")
    print(f"    [{model.num_queries + num_prefix}:] = spatial tokens ({x.shape[1] - model.num_queries - num_prefix} tokens)")
    print(f"    Total = {x.shape[1]} tokens")

# Test full forward
print("\n" + "=" * 80)
print("FULL FORWARD PASS TEST")
print("=" * 80)

model.eval()
with torch.no_grad():
    output = model.forward_frame(test_frame)

print(f"Output shapes:")
print(f"  pred_logits: {output['pred_logits'].shape} (expected: (1, num_queries, num_classes+1))")
print(f"  pred_masks: {output['pred_masks'].shape} (expected: (1, num_queries, H, W))")

class_logits = output['pred_logits']
pred_classes = class_logits.argmax(dim=-1)
print(f"\nPredicted classes: {pred_classes[0, :10]}")
print(f"Unique classes: {torch.unique(pred_classes).tolist()}")

if len(torch.unique(pred_classes)) == 1:
    print(f"\n⚠️  WARNING: Model predicting only class {pred_classes[0, 0]}")
    print("  This suggests spatial features are corrupted or not contributing to predictions")
