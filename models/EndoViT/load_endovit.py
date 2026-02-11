import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
segmentation_dir_path = os.path.join(
    current_dir,
    "finetuning",
    "semantic_segmentation",
    "model",
)
if not os.path.isdir(segmentation_dir_path):
    raise FileNotFoundError(
        "EndoViT model code not found at "
        f"{segmentation_dir_path}. Restore the 'finetuning/semantic_segmentation/model' "
        "folder under models/EndoViT before loading EndoViT."
    )
sys.path.insert(0, segmentation_dir_path)

from DPT.dpt.models import DPTSegmentationModel

def load_endovit(num_classes=12, repo_id="egeozsoy/EndoViT", filename="pytorch_model.bin"):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download EndoViT weights. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

    model = DPTSegmentationModel(
        num_classes,
        path=None,
        backbone="endovit_vitb16_224",
        readout="project",
        features=256,
        mae_hyperparams={"backbone_ckpt": weights_path},
    )

    return model

if __name__ == '__main__':
    

    model = load_endovit()

    random_input = torch.randn(16, 3, 224, 224)
    out = model(random_input)

    print(out.shape)
  
    for name, module in model.named_modules():
        print(name, module)
    
    # # Print keys
    # print('STATE DICT ENTRIES:')
    # for name, param in new_state_dict.items():
    #     print(name)
    # print('MODEL ENTRIES:')    
    # for name, param in model.named_parameters():
    #     print(name)