import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.surgenet import convnextv2
from models.surgenet import pvtv2
from models.surgenet import metaformer
from models.decoders.vit import ViTSegmenter, ViT
from models.GastroNet5M import vit_base_14


# This code loads all DINO model weights (v1, v2, v3) using timm library. Some unexpected keys are ignored during loading, this is intended behaviour.

# Optionally specify classes if needed for classifcation tasks, set to 0 for path embedding
n_classes = 0


def _load_surgenet2m_weights(model, weight_path):
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"SurgeNet2M weights not found at {weight_path}."
        )

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    if isinstance(state_dict, dict):
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
        if any(key.startswith("backbone.") for key in state_dict.keys()):
            state_dict = {
                key[len("backbone.") :]: value
                for key, value in state_dict.items()
                if key.startswith("backbone.")
            }

    target = model.backbone if hasattr(model, "backbone") else model

    if isinstance(state_dict, dict) and "pos_embed" in state_dict and hasattr(target, "pos_embed"):
        pos_embed = state_dict["pos_embed"]
        num_prefix = getattr(target, "num_prefix_tokens", 1)
        pos_prefix = pos_embed[:, :num_prefix]
        pos_grid = pos_embed[:, num_prefix:]
        grid_size_old = int(pos_grid.shape[1] ** 0.5)
        if hasattr(target.patch_embed, "grid_size"):
            grid_size_new = target.patch_embed.grid_size
        else:
            patch_size = target.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            img_size = target.img_size if hasattr(target, "img_size") else 224
            grid_val = img_size // patch_size
            grid_size_new = (grid_val, grid_val)
        pos_grid = pos_grid.reshape(1, grid_size_old, grid_size_old, -1).permute(0, 3, 1, 2)
        pos_grid = F.interpolate(
            pos_grid,
            size=grid_size_new,
            mode="bicubic",
            align_corners=False,
        )
        pos_grid = pos_grid.permute(0, 2, 3, 1).reshape(1, grid_size_new[0] * grid_size_new[1], -1)
        state_dict["pos_embed"] = torch.cat([pos_prefix, pos_grid], dim=1)

    msg = target.load_state_dict(state_dict, strict=False)
    return msg


###################################
### Loading dinov1(using timm) ###
###################################

# ViT-s
def load_dinov1_s():
    model = timm.create_model(
    'vit_small_patch16_224.dino',
    img_size=(224, 224),
    patch_size=16,
    num_classes=n_classes,
    )
    return model

# ViT-b
def load_dinov1_b():
    model = timm.create_model(
        'vit_base_patch16_224.dino',
        img_size=(224, 224),
        patch_size=16,
        num_classes=n_classes,
    )
    return model

###################################
### Loading dinov2 (using timm) ###
###################################

def load_dinov2_s(img_size=224):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    model = timm.create_model(
        'vit_small_patch14_dinov2',
        img_size=img_size,
        patch_size=14,
        num_classes=0,
    )
    return model

# ViT-b
def load_dinov2_b(img_size=224):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    model = timm.create_model(
        'vit_base_patch14_dinov2',
        img_size=img_size,
        patch_size=14,
        num_classes=0,
    )
    return model

# ViT-l
def load_dinov2_l(img_size=224):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    model = timm.create_model(
        'vit_large_patch14_dinov2',
        img_size=img_size,
        patch_size=14,
        num_classes=0,
    )
    return model


###########################################
### vit with linear head ###
###########################################
def load_lh_vit_s_dinov2(n_classes):
    vit = load_dinov2_s()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

def load_lh_vit_b_dinov2(n_classes):
    vit = load_dinov2_b()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

def load_lh_vit_l_dinov2(n_classes):
    vit = load_dinov2_l()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model


def load_dinov3_b():
    model = ViT(
        img_size=(256, 256),
        patch_size=16,
        backbone_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    print("\nLoaded DINOv3 ViT-Base (Hugging Face) weights.")
    return model


def load_dinov3_l():
    model = ViT(
        img_size=(256, 256),
        patch_size=16,
        backbone_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    print("\nLoaded DINOv3 ViT-Large (Hugging Face) weights.")
    return model


def load_lh_vit_b_dinov3(n_classes):
    vit = load_dinov3_b()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model


def load_lh_vit_l_dinov3(n_classes):
    vit = load_dinov3_l()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model



###########################################
### surgenet models ###
###########################################

def load_surgenet_convnextv2_tiny(num_classes=1):
    model = convnextv2.convnextv2_tiny_surgenet(num_classes=num_classes)
    return model

def load_surgenet_pvtv2_b2(num_classes=1):
    model = pvtv2.pvtv2_b2_surgenet(num_classes=num_classes)
    return model

def load_surgenet_caformer_s18(num_classes=1):
    model = metaformer.caformer_s18_surgenet(num_classes=num_classes)
    return model

def load_surgenetxl_caformer_s18(num_classes=1):
    model = metaformer.caformer_s18_surgenetxl(num_classes=num_classes)
    return model


###########################################
### EndoFM and EndoViT models ###
###########################################
# EndoFM and EndoViT load from local sources and Hugging Face weights.

def load_endofm(num_classes=1, device='cuda'):
    """Load EndoFM model for semantic segmentation."""
    from models.EndoFM.TransUNet.model import EndoFM
    model = EndoFM(num_classes=num_classes, device=device)
    return model

def load_endovit(num_classes=1):
    """Load EndoViT model for semantic segmentation."""
    from models.EndoViT.load_endovit import load_endovit
    model = load_endovit(num_classes=num_classes)
    return model


###########################################
### GastroNet5M model ###
###########################################

def load_lh_gastronet5m(n_classes):
    # Load GastroNet5M ViT base with patch size 14
    # Note: GastroNet5M was trained at 336x336 resolution, so img_size must be 336
    # Uses the actual DINO vit_base_14 function which loads gastronet weights
    vit = vit_base_14(patch_size=14, img_size=336, gastronet=True)
    
    # Wrap with ViTSegmenter for semantic segmentation
    # ViTSegmenter now handles both timm and DINO models
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model


###########################################
### SurgeNet2M-trained DINO models ###
###########################################

# DINOv1 - ViT-Base 224 (SurgeNet2M)
def load_dinov1_vitb_224_surgenet2m():
    model = timm.create_model(
        "vit_base_patch16_224_dino",
        img_size=(224, 224),
        patch_size=16,
        num_classes=0,
        pretrained=False,
    )
    weight_path = os.path.join(os.getcwd(), "weights", "DINOv1-vitb-224-SurgNet2M.pth")
    msg = _load_surgenet2m_weights(model, weight_path)
    print(f"\nLoaded DINOv1 ViT-Base 224 SurgeNet2M weights with msg:\n{msg}")
    return model

def load_lh_dinov1_vitb_224_surgenet2m(n_classes):
    vit = load_dinov1_vitb_224_surgenet2m()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

# DINOv2 - ViT-Base 336 (SurgeNet2M)
def load_dinov2_vitb_336_surgenet2m():
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        img_size=(336, 336),
        patch_size=14,
        num_classes=0,
        pretrained=False,
    )
    weight_path = os.path.join(os.getcwd(), "weights", "DINOv2-vitb-336-surgenet2M.pth")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    
    # Unwrap checkpoint if it has wrapper keys
    if isinstance(state_dict, dict):
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    
    # Strip common prefixes
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    if any(key.startswith("backbone.") for key in state_dict.keys()):
        state_dict = {key[len("backbone."):]: value for key, value in state_dict.items()}
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"\nLoaded DINOv2 ViT-Base 336 SurgeNet2M weights with msg:\n{msg}")
    return model

def load_lh_dinov2_vitb_336_surgenet2m(n_classes):
    vit = load_dinov2_vitb_336_surgenet2m()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

# DINOv3 - ViT-Large 256 (SurgeNet2M)
# Uses native DINOv3 architecture compatible with Meta-trained checkpoints
# Maps timm checkpoint weights to native DINOv3 format
def load_dinov3_vitl_256_surgenet2m():
    from models.dinov3_native import DINOv3ViT
    
    backbone = DINOv3ViT(
        img_size=256,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    )
    
    weight_path = os.path.join(os.getcwd(), "weights", "DINOv3-vitl-256-surgenet2M.pth")
    if os.path.isfile(weight_path):
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            if "teacher" in state_dict:
                state_dict = state_dict["teacher"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        # Strip common prefixes
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        if any(key.startswith("backbone.") for key in state_dict.keys()):
            state_dict = {key[len("backbone."):]: value for key, value in state_dict.items()}
        
        # Map timm weights to native DINOv3 format
        mapped_state_dict = {}
        for key, value in state_dict.items():
            # Map attention layers: attn.qkv -> attention.q/k/v_proj
            if ".attn.qkv.weight" in key:
                # Split qkv weights into separate q, k, v
                prefix = key.replace(".attn.qkv.weight", "")
                qkv_weight = value
                embed_dim = qkv_weight.shape[0] // 3
                q_proj, k_proj, v_proj = torch.chunk(qkv_weight, 3, dim=0)
                mapped_state_dict[f"{prefix}.attention.q_proj.weight"] = q_proj
                mapped_state_dict[f"{prefix}.attention.k_proj.weight"] = k_proj
                mapped_state_dict[f"{prefix}.attention.v_proj.weight"] = v_proj
            elif ".attn.qkv.bias" in key:
                prefix = key.replace(".attn.qkv.bias", "")
                qkv_bias = value
                embed_dim = qkv_bias.shape[0] // 3
                q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
                mapped_state_dict[f"{prefix}.attention.q_proj.bias"] = q_bias
                mapped_state_dict[f"{prefix}.attention.k_proj.bias"] = k_bias
                mapped_state_dict[f"{prefix}.attention.v_proj.bias"] = v_bias
            elif ".attn.proj.weight" in key:
                mapped_state_dict[key.replace(".attn.proj.weight", ".attention.o_proj.weight")] = value
            elif ".attn.proj.bias" in key:
                mapped_state_dict[key.replace(".attn.proj.bias", ".attention.o_proj.bias")] = value
            # Map layer scale: ls1.gamma -> layer_scale1.lambda1
            elif ".ls1.gamma" in key:
                mapped_state_dict[key.replace(".ls1.gamma", ".layer_scale1.lambda1")] = value
            elif ".ls2.gamma" in key:
                mapped_state_dict[key.replace(".ls2.gamma", ".layer_scale2.lambda1")] = value
            # Map MLP: fc1/fc2 -> up_proj/down_proj
            elif ".mlp.fc1.weight" in key:
                mapped_state_dict[key.replace(".mlp.fc1.weight", ".mlp.up_proj.weight")] = value
            elif ".mlp.fc1.bias" in key:
                mapped_state_dict[key.replace(".mlp.fc1.bias", ".mlp.up_proj.bias")] = value
            elif ".mlp.fc2.weight" in key:
                mapped_state_dict[key.replace(".mlp.fc2.weight", ".mlp.down_proj.weight")] = value
            elif ".mlp.fc2.bias" in key:
                mapped_state_dict[key.replace(".mlp.fc2.bias", ".mlp.down_proj.bias")] = value
            else:
                mapped_state_dict[key] = value
        
        msg = backbone.load_state_dict(mapped_state_dict, strict=False)
        print(f"\nLoaded DINOv3 ViT-Large 256 SurgeNet2M weights with msg:\n{msg}")
    
    # Wrap in ViT-compatible wrapper
    class ViTWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))
            self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))
        
        def forward(self, x):
            return self.backbone(x)
    
    return ViTWrapper(backbone)

def load_lh_dinov3_vitl_256_surgenet2m(n_classes):
    vit = load_dinov3_vitl_256_surgenet2m()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

# DINOv3 - ViT-Base 256 (SurgeNet2M)
# Uses native DINOv3 architecture compatible with Meta-trained checkpoints
# Maps timm checkpoint weights to native DINOv3 format
def load_dinov3_vitb_256_surgenet2m():
    from models.dinov3_native import DINOv3ViT
    
    backbone = DINOv3ViT(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    )
    
    weight_path = os.path.join(os.getcwd(), "weights", "DINOv3-vitb-256-surgenet2M.pth")
    if os.path.isfile(weight_path):
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            if "teacher" in state_dict:
                state_dict = state_dict["teacher"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        # Strip common prefixes
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
        if any(key.startswith("backbone.") for key in state_dict.keys()):
            state_dict = {key[len("backbone."):]: value for key, value in state_dict.items()}
        
        # Map timm weights to native DINOv3 format
        mapped_state_dict = {}
        for key, value in state_dict.items():
            # Map attention layers: attn.qkv -> attention.q/k/v_proj
            if ".attn.qkv.weight" in key:
                # Split qkv weights into separate q, k, v
                prefix = key.replace(".attn.qkv.weight", "")
                qkv_weight = value
                embed_dim = qkv_weight.shape[0] // 3
                q_proj, k_proj, v_proj = torch.chunk(qkv_weight, 3, dim=0)
                mapped_state_dict[f"{prefix}.attention.q_proj.weight"] = q_proj
                mapped_state_dict[f"{prefix}.attention.k_proj.weight"] = k_proj
                mapped_state_dict[f"{prefix}.attention.v_proj.weight"] = v_proj
            elif ".attn.qkv.bias" in key:
                prefix = key.replace(".attn.qkv.bias", "")
                qkv_bias = value
                embed_dim = qkv_bias.shape[0] // 3
                q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
                mapped_state_dict[f"{prefix}.attention.q_proj.bias"] = q_bias
                mapped_state_dict[f"{prefix}.attention.k_proj.bias"] = k_bias
                mapped_state_dict[f"{prefix}.attention.v_proj.bias"] = v_bias
            elif ".attn.proj.weight" in key:
                mapped_state_dict[key.replace(".attn.proj.weight", ".attention.o_proj.weight")] = value
            elif ".attn.proj.bias" in key:
                mapped_state_dict[key.replace(".attn.proj.bias", ".attention.o_proj.bias")] = value
            # Map layer scale: ls1.gamma -> layer_scale1.lambda1
            elif ".ls1.gamma" in key:
                mapped_state_dict[key.replace(".ls1.gamma", ".layer_scale1.lambda1")] = value
            elif ".ls2.gamma" in key:
                mapped_state_dict[key.replace(".ls2.gamma", ".layer_scale2.lambda1")] = value
            # Map MLP: fc1/fc2 -> up_proj/down_proj
            elif ".mlp.fc1.weight" in key:
                mapped_state_dict[key.replace(".mlp.fc1.weight", ".mlp.up_proj.weight")] = value
            elif ".mlp.fc1.bias" in key:
                mapped_state_dict[key.replace(".mlp.fc1.bias", ".mlp.up_proj.bias")] = value
            elif ".mlp.fc2.weight" in key:
                mapped_state_dict[key.replace(".mlp.fc2.weight", ".mlp.down_proj.weight")] = value
            elif ".mlp.fc2.bias" in key:
                mapped_state_dict[key.replace(".mlp.fc2.bias", ".mlp.down_proj.bias")] = value
            else:
                mapped_state_dict[key] = value
        
        msg = backbone.load_state_dict(mapped_state_dict, strict=False)
        print(f"\nLoaded DINOv3 ViT-Base 256 SurgeNet2M weights with msg:\n{msg}")
    
    # Wrap in ViT-compatible wrapper
    class ViTWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1))
            self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1))
        
        def forward(self, x):
            return self.backbone(x)
    
    return ViTWrapper(backbone)

def load_lh_dinov3_vitb_256_surgenet2m(n_classes):
    vit = load_dinov3_vitb_256_surgenet2m()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

