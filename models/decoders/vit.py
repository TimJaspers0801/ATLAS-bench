from typing import Optional
import timm
import torch
import torch.nn as nn
from models.decoders import build_decoder


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_dinov2",
        num_classes=0,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
        )

        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)


class ViTBackbone(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        
        # Handle both timm-wrapped models and DINO models
        if hasattr(vit_model, 'backbone'):
            # timm-wrapped ViT model
            backbone = vit_model.backbone
            self.is_dino = False
        else:
            # DINO model (direct DinoVisionTransformer)
            backbone = vit_model
            self.is_dino = True
        
        # Get patch size - handle both cases
        if hasattr(backbone.patch_embed, 'patch_size'):
            patch_size = backbone.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                self.patch_size = patch_size[0]
            else:
                self.patch_size = patch_size
        else:
            self.patch_size = backbone.patch_embed.patch_size[0]
            
        self.embed_dim = backbone.embed_dim
        
        # Get grid_size - handle both timm and DINO models
        if hasattr(backbone.patch_embed, 'grid_size'):
            # timm models have grid_size attribute
            self.grid_size = backbone.patch_embed.grid_size
        else:
            # DINO models - calculate from image size and patch size
            if hasattr(backbone, 'img_size'):
                img_size = backbone.img_size
                if isinstance(img_size, int):
                    grid_size_val = img_size // self.patch_size
                else:
                    grid_size_val = img_size[0] // self.patch_size
                self.grid_size = (grid_size_val, grid_size_val)
            else:
                # Fallback: calculate from positional embeddings
                num_patches = backbone.pos_embed.shape[1] - getattr(backbone, 'num_prefix_tokens', 1)
                grid_size_val = int(num_patches ** 0.5)
                self.grid_size = (grid_size_val, grid_size_val)
        
        self.backbone_ref = backbone

    def forward(self, x):
        backbone = self.backbone_ref
        
        # --- identical to ViT forward until tokens ---
        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)
        x = backbone.patch_drop(x)
        x = backbone.norm_pre(x)

        for blk in backbone.blocks:
            x = blk(x)

        x = backbone.norm(x)

        # --- remove CLS token ---
        if backbone.num_prefix_tokens > 0:
            x = x[:, backbone.num_prefix_tokens :, :]

        B, N, C = x.shape
        H, W = self.grid_size

        # --- tokens → feature map ---
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class ViTSegmenter(nn.Module):
    def __init__(
        self,
        vit_model,
        decoder_name: str,
        num_classes: int,
        img_size=(336, 336),
    ):
        super().__init__()

        self.backbone = ViTBackbone(vit_model)

        # Use patch_size from backbone which handles both timm and DINO models
        patch = self.backbone.patch_size
        upsample_factor = patch

        self.decoder = build_decoder(
            decoder_name,
            in_channels=self.backbone.embed_dim,
            num_classes=num_classes,
            upsample_factor=upsample_factor,
        )

    def forward(self, x):
        feats = self.backbone(x)      # (B, C, H_p, W_p)
        logits = self.decoder(feats)  # (B, num_classes, H, W)
        return logits



def get_vit_layer_id_from_name(name: str, num_blocks: int):
    """
    Assign a layer id to each parameter for LLRD.
    Higher id = closer to output = higher LR.
    """
    if name.startswith("decoder"):
        return num_blocks + 1  # decoder = highest LR

    if "backbone.vit.backbone.blocks" in name:
        # example: backbone.vit.backbone.blocks.11.attn.qkv.weight
        try:
            block_id = int(name.split("blocks.")[1].split(".")[0])
            return block_id + 1
        except Exception:
            pass

    if "backbone.vit.backbone.patch_embed" in name:
        return 0  # earliest layer

    if "backbone.vit.backbone.pos_embed" in name:
        return 0

    return 0


def get_param_groups_llrd_vit_segmenter(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    llrd_layer_decay: float = 0.8,
):
    """
    LLRD for ViTSegmenter:
    - Decoder gets base_lr
    - Last ViT blocks get higher LR
    - Early ViT layers get smaller LR
    """

    # Get reference to the actual backbone (handles both timm and DINO models)
    vit_backbone = model.backbone.backbone_ref
    num_blocks = len(vit_backbone.blocks)

    param_group_map = {}  # (layer_id, no_decay) -> params

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # ---- no weight decay rules ----
        no_decay = False
        lname = name.lower()

        if lname.endswith(".bias"):
            no_decay = True
        if (
            "norm" in lname
            or "layernorm" in lname
            or "ln" in lname
        ):
            no_decay = True
        if "pos_embed" in lname or "cls_token" in lname or "relative_position" in lname:
            no_decay = True

        layer_id = get_vit_layer_id_from_name(name, num_blocks)

        key = (layer_id, no_decay)
        param_group_map.setdefault(key, []).append(p)

    # ---- build param groups ----
    param_groups = []
    max_layer_id = max(k[0] for k in param_group_map.keys())

    for (layer_id, no_decay), params in sorted(param_group_map.items(), key=lambda x: x[0]):
        scale = llrd_layer_decay ** (max_layer_id - layer_id)
        lr = base_lr * scale

        param_groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
        })

    return param_groups
