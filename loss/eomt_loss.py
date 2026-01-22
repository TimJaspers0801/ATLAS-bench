from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)


class EoMTLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)

        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)


        return loss_total  # type: ignore

def convert_semantic_to_eomt_targets(sem_masks: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """
    Convert semantic segmentation masks into panoptic-like targets for EoMTLoss.

    Args:
        sem_masks: (B, H, W) tensor of semantic labels
        num_classes: number of valid classes
        ignore_index: label to ignore in mask (default 255, like in many datasets)

    Returns:
        List[dict]: Each dict has:
            - "masks": (N_i, H, W) binary masks for classes present in the image
            - "labels": (N_i,) tensor of class ids
    """
    B, H, W = sem_masks.shape
    targets = []

    for b in range(B):
        mask = sem_masks[b]
        labels = torch.unique(mask)
        labels = labels[(labels != ignore_index) & (labels < num_classes)]

        class_masks = [(mask == c).float() for c in labels]
        if len(class_masks) > 0:
            masks_tensor = torch.stack(class_masks)  # (N_i, H, W)
        else:
            masks_tensor = torch.zeros((0, H, W), dtype=torch.float, device=mask.device)

        targets.append({
            "masks": masks_tensor,
            "labels": labels
        })

    return targets
