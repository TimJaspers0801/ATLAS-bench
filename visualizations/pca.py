import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from models.load_models import load_dinov3_l, load_dinov3_b, load_dinov2_l, load_dinov2_b

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = load_dinov2_l()
model.to(device)
model.eval()

# Patch size
patch_size = model.patch_embed.patch_size
input_size = 336

# Preprocessing
def make_transform(resize_size: int = 336):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(
            mean=(0.430, 0.411, 0.296),
            std=(0.213, 0.156, 0.143)
        )
    ])

# Images
img_paths = [
    r'E:\SurgeSAM_final\cholecystectomy\J5bg8KTYrw0_ROBOT_TEST\clip_0001\images\frame_0988.jpg'
]

mask_paths = [
    r'E:\SurgeSAM_final\cholecystectomy\J5bg8KTYrw0_ROBOT_TEST\clip_0001\masks\frame_0988.png'
    ]

for img_path in img_paths:
    # Load original image
    image = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_paths[img_paths.index(img_path)]).convert('RGB')
    # image center crop
    width, height = image.size
    new_edge = min(width, height)
    left = (width - new_edge) / 2.5
    top = (height - new_edge) / 2.5
    right = (width + new_edge) / 2.5
    bottom = (height + new_edge) / 2.5
    image = image.crop((left, top, right, bottom))
    mask = mask.crop((left, top, right, bottom))

    # Preprocess for model
    image_tensor = make_transform(input_size)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract patch embeddings
        features = model.forward_features(image_tensor)  # [1, num_patches, embed_dim]
        features = features.squeeze(0)  # [num_patches, embed_dim]
        print("Features shape for PCA:", features.shape)

        if features.shape[0] < 3:
            raise ValueError("Not enough patch tokens for PCA. Check input image size.")

        # PCA to 3 components
        pca = PCA(n_components=3, whiten=True)
        pca_features = pca.fit_transform(features.cpu().numpy())

        # Normalize PCA features to 0-1
        norm_pca_feats = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())

        # Reshape into square image
        size = int(np.sqrt(norm_pca_feats.shape[0]))
        norm_pca_feats = norm_pca_feats[:size*size].reshape(size, size, 3)

        # Resize PCA to match original image size for overlay
        pca_image = Image.fromarray((norm_pca_feats * 255).astype(np.uint8)).resize(image.size)

        # Plot side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        # Original image left
        axes[0].imshow(image)
        axes[0].axis("off")

        # reference annotation
        axes[1].imshow(mask)  # overlay PCA
        axes[1].axis("off")


        # PCA plot right
        axes[2].imshow(pca_image, alpha=1)  # overlay PCA
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()