import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class VisionTransformerWithResize(nn.Module):
    def __init__(self, original_model, model_input_size=224):
        super(VisionTransformerWithResize, self).__init__()
        self.original_model = original_model
        self.model_input_size = model_input_size

    def forward(self, x):
        output_size = x.shape[-2:]
        x = F.interpolate(
            x,
            size=(self.model_input_size, self.model_input_size),
            mode='bilinear',
            align_corners=False,
        )
        x = self.original_model(x)
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return x

def EndoFM(num_classes=12, device='cuda:0'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/downstream/CVC-ClinicVideoDB/', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=9041, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--pretrained_model_weights', type=str, default='cvc.pth', help='pretrained weights')
    args = parser.parse_args([])


    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/pretrain/CVC-ClinicVideoDB/',
            'list_dir': './lists/lists_Synapse',
            'num_classes': num_classes,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    model = VisionTransformerWithResize(net, model_input_size=224).to(device)

    return model

if __name__ == '__main__':
    model = EndoFM(num_classes=12, device='cuda:0')
    inp = torch.randn(1, 3, 256, 256).to('cuda:0')
    out = model(inp)
    print(out.shape)
