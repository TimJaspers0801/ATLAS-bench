from typing import Optional
import timm
import torch
import torch.nn as nn
from transformers import AutoModel
from models.decoders import build_decoder


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        ckpt_path: Optional[str] = None,
        backbone_name="vit_large_patch14_reg4_dinov2",
        pixel_mean: Optional[list] = None,
        pixel_std: Optional[list] = None,
    ):
        super().__init__()

        if "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    backbone_name,
                    token=True,
                ),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        if pixel_mean is None:
            pixel_mean = [0.485, 0.456, 0.406]
        if pixel_std is None:
            pixel_std = [0.229, 0.224, 0.225]

        pixel_mean = torch.tensor(pixel_mean).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(pixel_std).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.embeddings.config.num_register_tokens + 1
        backbone.blocks = backbone.layer
        
        # Mark as converted HF model for detection in ViTBackbone
        backbone.is_hf_converted = True
        
        # Add forward_features method for compatibility with ViTBackbone
        def forward_features(x):
            outputs = backbone(pixel_values=x)
            return outputs.last_hidden_state
        
        backbone.forward_features = forward_features

        # Note: We do NOT delete embeddings or layer attributes because the
        # HF model's forward method still needs them. The model works fine
        # with both the timm-style attributes (patch_embed, blocks) and
        # the HF-style attributes (embeddings, layer) present.
        try:
            del backbone.patch_embed.mask_token
        except AttributeError:
            pass

        return backbone        


class ViTBackbone(nn.Module):
    """
    Simplified ViT backbone wrapper that uses timm's built-in methods.
    For timm models, uses forward_features() which handles everything correctly.
    For native DINOv3, uses custom forward pass.
    """
    def __init__(self, vit_model):
        super().__init__()
        
        # Determine model type
        if hasattr(vit_model, 'backbone'):
            # ViT class wrapper (from models/decoders/vit.py ViT class)
            self.backbone = vit_model.backbone
            self.is_vit_wrapper = True
        else:
            # Direct model (timm or native DINOv3)
            self.backbone = vit_model
            self.is_vit_wrapper = False
        
        # Check model type
        self.is_native_dinov3 = hasattr(self.backbone, 'prepare_tokens_with_masks')
        # Check for converted HF model (has marker attribute set in transformers_to_timm)
        self.is_hf_converted = hasattr(self.backbone, 'is_hf_converted')
        # HF model detection: has both embeddings and layer (transformers model, unconverted)
        self.is_hf_model = hasattr(self.backbone, 'embeddings') and hasattr(self.backbone, 'layer')
        
        # Get model properties
        if hasattr(self.backbone.patch_embed, 'patch_size'):
            patch_size = self.backbone.patch_embed.patch_size
            self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        else:
            self.patch_size = 16  # default
            
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        """Extract spatial features from ViT backbone."""
        if self.is_native_dinov3:
            # Native DINOv3: Use custom forward pass
            features = self.backbone(x)
            # GastroNet returns class logits (B, C) from forward(); use forward_features instead
            if features.dim() == 2 and hasattr(self.backbone, "forward_features"):
                features = self.backbone.forward_features(x)
                if isinstance(features, dict) and "x_norm_patchtokens" in features:
                    features = features["x_norm_patchtokens"]

            # Native DINOv3 returns (B, N, C) with prefix tokens already removed
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, C, H, W)
            return features
        elif self.is_hf_converted or self.is_hf_model:
            # HuggingFace model (converted or unconverted): Use the backbone's forward method with pixel_values argument
            outputs = self.backbone(pixel_values=x)
            features = outputs.last_hidden_state  # (B, N, C) with prefix tokens
            
            # Remove prefix tokens (CLS + register tokens)
            B, N, C = features.shape
            num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
            if num_prefix > 0:
                features = features[:, num_prefix:, :]
            
            # Reshape to spatial grid
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, C, H, W)
            return features
        else:
            # timm ViT or GastroNet: Use forward_features where available
            features = self.backbone.forward_features(x)

            # GastroNet forward_features returns a dict
            if isinstance(features, dict):
                if "x_norm_patchtokens" in features:
                    features = features["x_norm_patchtokens"]
                elif "x_prenorm" in features:
                    # Drop prefix tokens from pre-norm tokens
                    tokens = features["x_prenorm"]
                    num_prefix = getattr(self.backbone, "num_register_tokens", 0) + 1
                    features = tokens[:, num_prefix:, :]
                else:
                    raise ValueError("Unsupported GastroNet feature dict keys")

            # forward_features returns (B, N, C) where N includes prefix tokens
            B, N, C = features.shape

            # Remove prefix tokens (CLS + register tokens)
            num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
            if num_prefix > 0:
                features = features[:, num_prefix:, :]

            # Reshape to spatial grid
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, C, H, W)

            return features

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

    # Match blocks in the backbone (handles both old and new structure)
    if "backbone.backbone.blocks" in name or "backbone.vit.backbone.blocks" in name:
        # example: backbone.backbone.blocks.11.attn.qkv.weight
        try:
            block_id = int(name.split("blocks.")[1].split(".")[0])
            return block_id + 1
        except Exception:
            pass

    # Match patch_embed or pos_embed (earliest layers)
    if "patch_embed" in name or "pos_embed" in name:
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

    # Get reference to the actual backbone
    vit_backbone = model.backbone.backbone
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
