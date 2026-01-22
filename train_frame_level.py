import argparse
import wandb
import torch
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets.atlas import AtlasDataset
import torchvision.transforms.v2 as T
from torch import nn
from utils import load_checkpoint, bgr_palette
import pandas as pd

from models.load_models import load_eomt_s_dinov2, load_eomt_b_dinov2, load_eomt_l_dinov2,\
                               load_eomt_s_dinov3, load_eomt_b_dinov3, load_eomt_l_dinov3, \
                               load_surgenet_caformer_s18, load_surgenet_convnextv2_tiny, load_surgenet_pvtv2_b2, load_surgenetxl_caformer_s18

from models.eomt.eomt import get_param_groups_llrd
from evaluation.dataset_evaluation import evaluate_model
from evaluation.visual_logging import collect_visual_grids


def train(args):
    # set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Init wandb
    wandb.init(project="Atlas",
               config=vars(args),
               name=args.experiment_name,
               dir=r"C:\Users\20172619\wandb\wandb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Define transforms
    train_transform = T.Compose([
        T.RandomResizedCrop(
            args.img_size,
            scale=(0.8, 1.0),
            interpolation=T.InterpolationMode.BICUBIC
        ),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
    ])

    val_transform = T.Compose([
        T.Resize(args.img_size, interpolation=T.InterpolationMode.NEAREST),
        T.CenterCrop(args.img_size),
    ])


    # Create datasets and loaders
    train_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="train",
        transform=train_transform,
        first_frame_only=args.first_frame_only,
        frame_percentage=args.frame_percentage,
    )

    val_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="val",
        transform=val_transform,
    )

    test_dataset = AtlasDataset(
        zip_path=args.data_path,
        split="test",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT for clip evaluation
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT for clip evaluation
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Training on {len(train_dataset)} samples, Validation on {len(val_dataset)} samples. Testing on {len(test_dataset)}.")

    # Model
    if 'eomt' in args.model:
        base_lr = 1e-4
        weight_decay = 0.05
        llrd_factor = 0.8

    if args.model == 'eomt-s-dinov2':
        model = load_eomt_s_dinov2(n_classes=args.num_classes, num_q=args.num_classes)
    elif args.model == 'eomt-b-dinov2':
        model = load_eomt_b_dinov2(n_classes=args.num_classes, num_q=args.num_classes)
    elif args.model == 'eomt-l-dinov2':
        model = load_eomt_l_dinov2(n_classes=args.num_classes, num_q=args.num_classes)
    elif args.model == 'eomt-s-dinov3':
        model = load_eomt_s_dinov3(n_classes=args.num_classes, num_q=args.num_classes)
    elif args.model == 'eomt-b-dinov3':
        model = load_eomt_b_dinov3(n_classes=args.num_classes, num_q=args.num_classes)
    elif args.model == 'eomt-l-dinov3':
        model = load_eomt_l_dinov3(n_classes=args.num_classes, num_q=args.num_classes)
    elif args.model == 'convnextv2':
        model = load_surgenet_convnextv2_tiny(num_classes=args.num_classes)
    elif args.model == 'caformer':
        model = load_surgenet_caformer_s18(num_classes=args.num_classes)
    elif args.model == 'pvtv2':
        model = load_surgenet_pvtv2_b2(num_classes=args.num_classes)
    elif args.model == 'surgenetxl':
        model = load_surgenetxl_caformer_s18(num_classes=args.num_classes)
    else:
        print(f"Warning: Model {args.model} not recognized.")

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    model.to(device)


    if 'eomt' in args.model.lower():
        from loss.eomt_loss import EoMTLoss
        train_criterion = EoMTLoss(num_points=12544,
                            oversample_ratio=3.0,
                            importance_sample_ratio=0.75,
                            mask_coefficient=5.0,
                            dice_coefficient=5.0,
                            class_coefficient=2.0,
                            num_labels=args.num_classes,
                            no_object_coefficient=1.0,
                            ).to(device)
        param_groups = get_param_groups_llrd(model, base_lr=base_lr, weight_decay=weight_decay,
                                             llrd_layer_decay=llrd_factor)


    criterion = nn.CrossEntropyLoss(ignore_index=255)


    if 'eomt' in args.model.lower():
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create outputs folder
    output_path = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_path, exist_ok=True)

    best_dice = 0.0
    # checkpoint dir
    checkpoint = os.listdir(output_path)
    if checkpoint:
        # find the last saved model
        last_checkpoint = sorted(checkpoint, key=lambda x: os.path.getmtime(os.path.join(output_path, x)))[-1]
        best_dice = float(last_checkpoint.split('_')[-1].replace('.pt', ''))
        best_epoch = int(last_checkpoint.split('_')[3])
        print(f"Resuming from last checkpoint: {last_checkpoint} with dice {best_dice:.4f}")
        best_model_path = os.path.join(
            output_path,
            f"best_model_epoch_{best_epoch}_dice_{best_dice:.4f}.pt"
        )
        print(best_model_path)
    else:
        print("No previous checkpoints found.")
        best_model_path = None  # ← track the path of the last saved best model
        best_epoch = None

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            images, masks = images.to(device), masks.to(device).squeeze()

            if 'eomt' in args.model.lower():
                mask_logits_per_layer, class_logits_per_layer = model(images, return_semantic=False)
                from loss.eomt_loss import convert_semantic_to_eomt_targets
                seg_targets = convert_semantic_to_eomt_targets(masks, num_classes=7)  # list of dicts

                # accumulate scalar losses across layers
                seg_loss = None
                n_layers = len(mask_logits_per_layer)

                for i, (mask_logits, class_logits) in enumerate(zip(mask_logits_per_layer, class_logits_per_layer)):
                    # slice logits to only those images that have masks
                    # call EoMT/M2F loss to get dict of losses for this layer
                    losses_dict = train_criterion(
                        masks_queries_logits=mask_logits,
                        class_queries_logits=class_logits,
                        targets=seg_targets,
                    )
                    # reduce this layer's losses to a single scalar using the module helper
                    layer_loss = train_criterion.loss_total(losses_dict)  # scalar tensor

                    loss = layer_loss if seg_loss is None else seg_loss + layer_loss
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if step % args.log_interval == 0:
                wandb.log({"Train Loss": loss.item(), "Epoch": epoch})

            pbar.set_postfix({"Loss": loss.item()})

        # Learning rate scheduler step
        learning_rate_scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        wandb.log({"Avg Train Loss": avg_train_loss, "Epoch": epoch})
        print(f"[Epoch {epoch}] Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation ---
        metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        wandb.log({
            "Val mIoU": metrics["mIoU"],
            "Val Dice": metrics["Dice"],
            "Val AP": metrics["AP"],
            "Val AP50": metrics["AP50"],
            "Val AP75": metrics["AP75"],
            "Epoch": epoch,
        })

        # --- qualitative ---
        if epoch % 5 == 0:
            print("Collecting visual grids for wandb...")
            grids = collect_visual_grids(
                model=model,
                dataloader=val_loader,
                device=device,
                palette=bgr_palette,
                mean=val_loader.dataset.mean,
                std=val_loader.dataset.std,
            )

            for i, grid in enumerate(grids):
                wandb.log({
                    f"Val clip {i}": wandb.Image(
                        grid,
                        caption="Image | GT overlay | Prediction overlay"
                    )
                }, step=epoch)

        current_score = metrics["Dice"]

        if current_score > best_dice:
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_dice = current_score
            best_epoch = epoch

            best_model_path = os.path.join(
                output_path,
                f"best_model_epoch_{epoch}_dice_{current_score:.4f}.pt"
            )

            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics
            }, best_model_path)

            print(f"✅ Saved new best model to {best_model_path}")

    # After training loop ends, do final evaluation
    if best_model_path is None:
        print("⚠️ No best model was ever saved. Skipping evaluation.")
        return

    print(f"Loading best model from {best_model_path} for final evaluation.")
    load_checkpoint(model, best_model_path)
    print("Evaluating on Validation set...")
    val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
        )
    print("Evaluating on Test set...")
    test_metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
        )
    # Log final results
    wandb.log({
        "Final Val mIoU": val_metrics["mIoU"],
        "Final Val Dice": val_metrics["Dice"],
        "Final Val AP": val_metrics["AP"],
        "Final Val AP50": val_metrics["AP50"],
        "Final Val AP75": val_metrics["AP75"],
        "Final Test mIoU": test_metrics["mIoU"],
        "Final Test Dice": test_metrics["Dice"],
        "Final Test AP": test_metrics["AP"],
        "Final Test AP50": test_metrics["AP50"],
        "Final Test AP75": test_metrics["AP75"],
    })

    # save results to excel
    results_df = pd.DataFrame({
        "Metric": ["mIoU", "Dice", "AP", "AP50", "AP75"],
        "Validation": [val_metrics["mIoU"], val_metrics["Dice"], val_metrics["AP"], val_metrics["AP50"], val_metrics["AP75"]],
        "Test": [test_metrics["mIoU"], test_metrics["Dice"], test_metrics["AP"], test_metrics["AP50"], test_metrics["AP75"]],
    })
    results_df.to_excel(os.path.join(output_path, "final_evaluation.xlsx"), index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to SurgeNet pre-trained weights")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=46)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--visual_log", type=int, default=50, help="Log images every N steps")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of this run")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save models and logs")
    parser.add_argument("--url", type=str, default=None, help="URL to pretrained weights (SurgeNet or others)")
    parser.add_argument("--first_frame_only", action="store_true", help="Use only the first frame of each clip")
    parser.add_argument('--frame_percentage', type=int, default=100, help='Percentage of frames to use')
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    train(args)