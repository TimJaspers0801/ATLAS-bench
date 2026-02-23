from typing import Optional
import inspect
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
    ):
        super().__init__()

        if "/" in backbone_name:
            # Check if ckpt_path points to a .pth file (raw weights)
            if ckpt_path and ckpt_path.endswith('.pth'):
                # Create model from pretrained config (downloads config only)
                hf_model = AutoModel.from_pretrained(backbone_name)
                
                # Load the custom weights
                state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                # Handle different state dict formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']

                # Fix key naming to match HuggingFace format
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key
                    
                    # Remove 'backbone.' prefix if present
                    if new_key.startswith('backbone.'):
                        new_key = new_key.replace('backbone.', '', 1)
                    
                    # Rename to match HF naming convention
                    new_key = new_key.replace('patch_embed.', 'embeddings.')
                    new_key = new_key.replace('blocks.', 'layer.')
                    
                    # Skip pixel_mean and pixel_std (you register these separately)
                    if new_key not in ['pixel_mean', 'pixel_std']:
                        new_state_dict[new_key] = value
                
                # Load weights into the HF model
                missing_keys, unexpected_keys = hf_model.load_state_dict(new_state_dict, strict=False)
                
                if missing_keys:
                    print(f"[INFO] Missing keys when loading checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"[INFO] Unexpected keys when loading checkpoint: {unexpected_keys}")

                # Convert to timm format
                self.backbone = self.transformers_to_timm(hf_model, img_size)
            else:
                # Original path: load from HF hub
                self.backbone = self.transformers_to_timm(
                    AutoModel.from_pretrained(
                        backbone_name,
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

            self.backbone.patch_size = patch_size
            self._orig_gil = self.backbone.get_intermediate_layers
            self.backbone.get_intermediate_layers = self._get_intermediate_layers_timm

        # --- Normalization buffers (conditional on checkpoint) ---
        if ckpt_path is not None:
            # Custom stats for checkpointed models (SurgeNet)
            mean = [0.46888983, 0.29536288, 0.28712815]
            std  = [0.24689102, 0.21034359, 0.21188641]
        else:
            # Default (ImageNet)
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]
            
        pixel_mean = torch.tensor(mean, dtype=torch.float32).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(std,  dtype=torch.float32).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
    
    def _get_intermediate_layers_timm(self, x, n, **_):
        return self._orig_gil(x, n, return_prefix_tokens=True)

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_size = backbone.embeddings.config.patch_size
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer
        backbone.get_intermediate_layers = self._get_intermediate_layers_hf

        # Only delete mask_token, keep embeddings and layer for HF model compatibility
        try:
            del backbone.patch_embed.mask_token
        except AttributeError:
            pass

        return backbone

    def _get_intermediate_layers_hf(self, x, n, return_class_token=True, **kwargs):
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden = out.hidden_states
        idxs = list(n)
        picks = []
        npt = self.backbone.num_prefix_tokens
        for idx in idxs:
            hs = hidden[idx]
            cls_tok = hs[:, :1, :]
            patch_tok = hs[:, npt:, :]
            picks.append((patch_tok, cls_tok) if return_class_token else patch_tok)
        return picks        



class ViTBackbone(nn.Module):
    """
    ViT backbone wrapper that extracts patch features from the model.
    Handles both timm and HuggingFace ViT models.
    """
    def __init__(self, vit_model):
        super().__init__()
        
        # Get the backbone (either from ViT wrapper or directly)
        if hasattr(vit_model, 'backbone'):
            self.backbone = vit_model.backbone
            # Check if ViT wrapper has custom get_intermediate_layers (for HF models)
            if hasattr(vit_model, '_get_intermediate_layers_hf'):
                # Bind the HF wrapper to this instance
                self.get_intermediate_layers_fn = lambda x, n: vit_model._get_intermediate_layers_hf(x, n, return_class_token=False)
                self.is_hf_model = True
            elif hasattr(vit_model, '_get_intermediate_layers_timm'):
                # Bind the timm wrapper to this instance
                self.get_intermediate_layers_fn = lambda x, n: vit_model._get_intermediate_layers_timm(x, n, return_prefix_tokens=False)
                self.is_hf_model = False
            else:
                # Direct model (could be timm or GastroNet/DINO)
                # Check the method signature to determine the correct parameter name
                sig = inspect.signature(self.backbone.get_intermediate_layers)
                if 'return_prefix_tokens' in sig.parameters:
                    # timm model
                    self.get_intermediate_layers_fn = lambda x, n: self.backbone.get_intermediate_layers(x, n, return_prefix_tokens=False)
                    self.is_hf_model = False
                elif 'return_class_token' in sig.parameters:
                    # GastroNet (DINO) model
                    self.get_intermediate_layers_fn = lambda x, n: self.backbone.get_intermediate_layers(x, n, return_class_token=False)
                    self.is_hf_model = False
                else:
                    # Fallback: call without any prefix parameter
                    self.get_intermediate_layers_fn = lambda x, n: self.backbone.get_intermediate_layers(x, n)
                    self.is_hf_model = False
        else:
            self.backbone = vit_model
            # Check if it's an HF model (has embeddings and layer attributes)
            if hasattr(self.backbone, 'embeddings') and hasattr(self.backbone, 'layer'):
                self.get_intermediate_layers_fn = lambda x, n: self.backbone.get_intermediate_layers(x, n, return_class_token=False)
                self.is_hf_model = True
            else:
                # Check the method signature to determine the correct parameter name
                sig = inspect.signature(self.backbone.get_intermediate_layers)
                if 'return_prefix_tokens' in sig.parameters:
                    # timm model
                    self.get_intermediate_layers_fn = lambda x, n: self.backbone.get_intermediate_layers(x, n, return_prefix_tokens=False)
                    self.is_hf_model = False
                elif 'return_class_token' in sig.parameters:
                    # GastroNet (DINO) model
                    self.get_intermediate_layers_fn = lambda x, n: self.backbone.get_intermediate_layers(x, n, return_class_token=False)
                    self.is_hf_model = False
                else:
                    # Fallback: call without any prefix parameter
                    self.get_intermediate_layers_fn = lambda x, n: self.backbone.get_intermediate_layers(x, n)
                    self.is_hf_model = False
        
        # Get model properties
        if hasattr(self.backbone, 'patch_embed'):
            patch_size = self.backbone.patch_embed.patch_size
            self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        elif hasattr(self.backbone, 'patch_size'):
            self.patch_size = self.backbone.patch_size
        else:
            self.patch_size = 16  # default
            
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        """Extract spatial patch features from ViT backbone."""
        # Use get_intermediate_layers to extract patch features from the last layer
        # This works for both timm and HF models thanks to our wrapper
        intermediate = self.get_intermediate_layers_fn(x, [11])
        
        # intermediate is a list of outputs from each requested layer
        # For layer 11 (the last transformer layer), get the patch tokens
        patch_tokens = intermediate[-1]  
        
        # patch_tokens should be (B, N, C) where N is number of patches
        # If it's a tuple (patch_tok, cls_tok), take just the patches
        if isinstance(patch_tokens, tuple):
            patch_tokens = patch_tokens[0]
        
        # Reshape to spatial grid (B, N, C) -> (B, C, H, W)
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        features = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        
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
