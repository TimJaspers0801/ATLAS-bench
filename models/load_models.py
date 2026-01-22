import torch
import timm
from models.eomt import eomt, vit
from models.surgenet import convnextv2
from models.surgenet import pvtv2
from models.surgenet import metaformer


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
    model = vit.ViT(
        backbone_name='vit_small_patch14_dinov2',
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
    model = vit.ViT(
        backbone_name='vit_base_patch14_dinov2',
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
    model = vit.ViT(
        backbone_name='vit_large_patch14_dinov2',
        img_size=(336, 336),
        patch_size=14,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov2_vitl'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv2 ViT-l weights with msg:\n", msg)
    return model

###########################################
### Loading dinov3 (using transformers) ###
###########################################

# ViT-s
def load_dinov3_s():
    model = timm.create_model(
        'vit_small_patch16_dinov3',
        img_size=(336, 336),
        patch_size=16,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov3_vits'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv3 ViT-s weights with msg:\n", msg)
    return model

# ViT-b
def load_dinov3_b():
    model = timm.create_model(
        'vit_base_patch16_dinov3',
        img_size=(336, 336),
        patch_size=16,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov3_vitb'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv3 ViT-b weights with msg:\n", msg)
    return model

# ViT-l
def load_dinov3_l():
    model = timm.create_model(
        'vit_large_patch16_dinov3',
        img_size=(336, 336),
        patch_size=16,
        num_classes=n_classes,
    )
    state_dict = torch.hub.load_state_dict_from_url(urls['path_dinov3_vitl'])
    msg = model.load_state_dict(state_dict, strict=False)
    print("\nLoaded DINOv3 ViT-l weights with msg:\n", msg)
    return model

###########################################
### Loading eomt models ###
###########################################

def load_eomt_s_dinov2(n_classes, num_q=50):
    model = eomt.eomt_vits_dinov2(num_classes=n_classes, num_q=num_q)
    return model

def load_eomt_b_dinov2(n_classes, num_q=50):
    model = eomt.eomt_vitb_dinov2(num_classes=n_classes, num_q=num_q)
    return model

def load_eomt_l_dinov2(n_classes, num_q=50):
    model = eomt.eomt_vitl_dinov2(num_classes=n_classes, num_q=num_q)
    return model

def load_eomt_s_dinov3(n_classes, num_q=50):
    model = eomt.eomt_vits_dinov3(num_classes=n_classes, num_q=num_q)
    return model

def load_eomt_b_dinov3(n_classes, num_q=50):
    model = eomt.eomt_vitb_dinov3(num_classes=n_classes, num_q=num_q)
    return model

def load_eomt_l_dinov3(n_classes, num_q=50):
    model = eomt.eomt_vitl_dinov3(num_classes=n_classes, num_q=num_q)
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

