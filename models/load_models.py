import torch
import timm
import os
from models.surgenet import convnextv2
from models.surgenet import pvtv2
from models.surgenet import metaformer
from models.decoders.vit import ViTSegmenter
from models.GastroNet5M import vit_base_14
from models.dinov1 import vision_transformer as dinov1_vit
from models.dinov2 import vision_transformer as dinov2_vit
from models.dinov3 import vision_transformer as dinov3_vit


# This code loads all DINO model weights (v1, v2, v3) using timm library. Some unexpected keys are ignored during loading, this is intended behaviour.

# Optionally specify classes if needed for classifcation tasks, set to 0 for path embedding
n_classes = 0

# URLs for pre-trained DINO weights
urls = {
    'path_dinov1_vits': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTs16_size224_SurgeNetXL.pth?download=true',
    'path_dinov1_vitb': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv1_ViTb16_size224_SurgeNetXL.pth?download=true',

    'path_dinov2_vits': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTs14_size336_SurgeNetXL.pth?download=true',
    'path_dinov2_vitb': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTb14_size336_SurgeNetXL.pth?download=true',
    'path_dinov2_vitl': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTl14_size336_SurgeNetXL.pth?download=true',

    'path_dinov3_vits': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTs16_size336_SurgeNetXL.pth?download=true',
    'path_dinov3_vitb': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTb16_size336_SurgeNetXL.pth?download=true',
    'path_dinov3_vitl': 'https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv3_ViTl16_size336_SurgeNetXL.pth?download=true',
}


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
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov1_vits'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv1 ViT-s weights with msg:\n", msg)
    return model

# ViT-b
def load_dinov1_b():
    model = timm.create_model(
        'vit_base_patch16_224.dino',
        img_size=(224, 224),
        patch_size=16,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov1_vitb'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv1 ViT-b weights with msg:\n", msg)
    return model

###################################
### Loading dinov2 (using timm) ###
###################################

def load_dinov2_s():
    # ViT-s
    model = timm.create_model(
        'vit_small_patch14_dinov2',
        img_size=(336, 336),
        patch_size=14,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov2_vits'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv2 ViT-s weights with msg:\n", msg)
    return model

# ViT-b
def load_dinov2_b():
    model = timm.create_model(
        'vit_base_patch14_dinov2',
        img_size=(336, 336),
        patch_size=14,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov2_vitb'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv2 ViT-b weights with msg:\n", msg)
    return model

# ViT-l
def load_dinov2_l():
    model = timm.create_model(
        'vit_large_patch14_dinov2',
        img_size=(336, 336),
        patch_size=14,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov2_vitl'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv2 ViT-l weights with msg:\n", msg)
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
    model = dinov1_vit.vit_base(patch_size=16, num_classes=0, img_size=224)
    weight_path = os.path.join(os.getcwd(), 'weights', 'DINOv1-vitb-224-SurgNet2M.pth')
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(state_dict, strict=False)
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
    model = dinov2_vit.vit_base_14(patch_size=14, num_register_tokens=0)
    weight_path = os.path.join(os.getcwd(), 'weights', 'DINOv2-vitb-336-surgenet2M.pth')
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
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
def load_dinov3_vitl_256_surgenet2m():
    model = dinov3_vit.vit_large(patch_size=16, img_size=256, num_classes=0)
    weight_path = os.path.join(os.getcwd(), 'weights', 'DINOv3-vitl-256-surgenet2M.pth')
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"\nLoaded DINOv3 ViT-Large 256 SurgeNet2M weights with msg:\n{msg}")
    return model

def load_lh_dinov3_vitl_256_surgenet2m(n_classes):
    vit = load_dinov3_vitl_256_surgenet2m()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

# DINOv3 - ViT-Small 256 (SurgeNet2M) - Placeholder for future weights
def load_dinov3_vits_256_surgenet2m():
    model = dinov3_vit.vit_small(patch_size=16, img_size=256, num_classes=0)
    weight_path = os.path.join(os.getcwd(), 'weights', 'DINOv3-vits-256-surgenet2M.pth')
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"\nLoaded DINOv3 ViT-Small 256 SurgeNet2M weights with msg:\n{msg}")
    else:
        print(f"\nWarning: Weight file not found at {weight_path}. Using uninitialized model.")
    return model

def load_lh_dinov3_vits_256_surgenet2m(n_classes):
    vit = load_dinov3_vits_256_surgenet2m()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

# DINOv3 - ViT-Base 256 (SurgeNet2M) - Placeholder for future weights
def load_dinov3_vitb_256_surgenet2m():
    model = dinov3_vit.vit_base(patch_size=16, img_size=256, num_classes=0)
    weight_path = os.path.join(os.getcwd(), 'weights', 'DINOv3-vitb-256-surgenet2M.pth')
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"\nLoaded DINOv3 ViT-Base 256 SurgeNet2M weights with msg:\n{msg}")
    else:
        print(f"\nWarning: Weight file not found at {weight_path}. Using uninitialized model.")
    return model

def load_lh_dinov3_vitb_256_surgenet2m(n_classes):
    vit = load_dinov3_vitb_256_surgenet2m()
    model = ViTSegmenter(
        vit_model=vit,
        decoder_name="linear",
        num_classes=n_classes)
    return model

