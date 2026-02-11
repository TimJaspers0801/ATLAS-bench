import torch
import json
import os
import sys

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

import src.util as util
from DPT.dpt.models import DPTSegmentationModel




def _init_model(config, state_dict=None):
    util.header(title="Model", separator="*")

    ############################### Setup ###############################

    DPT_hyperparams = config["Model"]["DPT Hyperparams"]
    MAE_hyperparams = config["Model"]["MAE Hyperparams"]

    DPT_kwargs = {
        "readout": DPT_hyperparams["readout"],
        "features": DPT_hyperparams["features"],
        "use_bn": DPT_hyperparams["use_bn"],
    }

    ############################# Load Model ############################

    model= DPTSegmentationModel(
        DPT_hyperparams["num_classes"],
        path = None, # We don't wish to load any saved checkpoints here. Checkpoint loading is done in the trainer.
        backbone = DPT_hyperparams["backbone"],
        mae_hyperparams=MAE_hyperparams,
        # unnecessary hyperparams, because these are set by default
        **DPT_kwargs
    )

    print("", end="\n\n")
    util.header(title="Full Model", total_length=50)

    print(model)
    
    if (state_dict):
        util.new_section()
        model.load_state_dict(state_dict)
        print("Loaded model state dict!")
        
        # Print keys
        print('STATE DICT ENTRIES:')
        for name, param in state_dict:
            print(name)
        print('MODEL ENTRIES:')    
        for name, param in model.named_parameters():
            print(name)
        
        
  
    return model

def load_pretrained_checkpoint(config):
        util.header(title="Loading Checkpoint", separator="*")

        # Resume training from checkpoint.
        checkpoint = {}
        resume_from_checkpoint = config["General Hyperparams"]["resume_training"]
        load_checkpoint = config["General Hyperparams"]["load_checkpoint"]

        # There are two options:
        #   1) resume_from_checkpoint
        #       -> Will continue the training from where it left off.
        #       -> Last epoch, model_state_dict, optimizer_state_dict,
        #          scheduler_state_dict will be loaded. If performance
        #          optimizations are enabled, scaler_state_dict will
        #          also be loaded.
        #
        #   2) load_checkpoint
        #       -> Will start training from scratch, but will load a
        #          pretrained model_state_dict.
        #
        # NOTE: Resume_from_checkpoint has priority over load_checkpoint

        if (resume_from_checkpoint):
            checkpoint = torch.load(resume_from_checkpoint)

        elif(load_checkpoint):
            checkpoint = torch.load(load_checkpoint)

            checkpoint["epoch"] = 0
            checkpoint["optimizer"] = None
            checkpoint["scheduler"] = None
            checkpoint["scaler"] = None

        # If loading from a pretrained checkpoint, then mae_ckpt should not be loaded.
        if (checkpoint):
            print(f"Checkpoint loaded from: {resume_from_checkpoint if resume_from_checkpoint else load_checkpoint}")
            print(f"\t -> Setting \"mae_ckpt\" = \"\".")
            config["Model"]["MAE Hyperparams"]["mae_ckpt"] = ""
        else:
            print("No checkpoint loaded!")

        return  checkpoint


def get_config(config_path):

    #assert config_path.is_file(), "Please provide a valid path to a \".json\" config file."

    with open(config_path, "r") as read_file:
        config = json.load(read_file)   
        return config

def EndoViT(num_classes=12, config_path=None, weights_path=None):
        repo_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
        weights_dir = os.path.join(repo_root, "weights")
        config_path = config_path or os.path.join(weights_dir, "endovit_config.json")
        weights_path = weights_path or os.path.join(weights_dir, "endovit_seg.pth")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                "EndoViT config not found. Expected config at "
                f"{config_path}."
            )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "EndoViT weights not found. Expected weights at "
                f"{weights_path}."
            )

        # Load Config file
        config = get_config(config_path)
        config["Model"]["DPT Hyperparams"]["num_classes"] = num_classes  # Update to desired number
        config["Model"]["MAE Hyperparams"]["nb_classes"] = num_classes   # Make consistent across parameters
        checkpoint = load_pretrained_checkpoint(config)
        

        # Load Model
        model_state_dict = checkpoint.get("model")
        model = _init_model(config, state_dict=model_state_dict)
        
        # Load state_dict into model after small modifications
        state_dict = torch.load(weights_path)
        state_dict = state_dict['model']
        new_state_dict = {f"pretrained.model.{key}": value for key, value in state_dict.items()}
        out = model.load_state_dict(new_state_dict, strict=False)
        print(out)

        return model

if __name__ == '__main__':
    

    model = EndoViT()

    random_input = torch.randn(16, 3, 256, 256)
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