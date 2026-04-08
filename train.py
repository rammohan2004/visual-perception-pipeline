"""Training entrypoint
"""

"""Training entrypoint
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

def get_transforms():
    """Returns train and validation albumentations transforms."""
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    return train_transform, val_transform

def apply_transfer_strategy(model, strategy):
    """Applies the freezing logic to the model's encoder."""
    if strategy == "strict":
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == "partial":
        for param in model.encoder.block1.parameters(): param.requires_grad = False
        for param in model.encoder.block2.parameters(): param.requires_grad = False
        for param in model.encoder.block3.parameters(): param.requires_grad = False
    # If "full", we do nothing (default is requires_grad=True)

def main():
    parser = argparse.ArgumentParser(description="Train DA6401 Assignment 2 Models")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "localization", "segmentation"])
    parser.add_argument("--transfer_strategy", type=str, default="full", choices=["strict", "partial", "full"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--use_batchnorm", action="store_true")
    parser.add_argument("--data_dir", type=str, default="./dataset")
    args = parser.parse_args()

    wandb.init(project="da6401_assignment_2", name=f"{args.task}_{args.transfer_strategy}_bn{args.use_batchnorm}_drop{args.dropout_p}", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, val_transform = get_transforms()
    
    # TODO: Initialize train_dataset and train_loader here
    # =========================
    # Dataset + Loader
    # =========================
    train_dataset = OxfordIIITPetDataset(
        root_dir=args.data_dir,
        split="train",
        transforms=train_transform
    )

    val_dataset = OxfordIIITPetDataset(
        root_dir=args.data_dir,
        split="test",
        transforms=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # Initialize Model and Loss
    if args.task == "classification":
        model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)
        criterion = nn.CrossEntropyLoss()
        save_name = "classifier.pth"
        
    elif args.task == "localization":
        model = VGG11Localizer(dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)
        mse_loss = nn.MSELoss()
        iou_loss = IoULoss(reduction="mean")
        
        # Define scaling weights
        # Since MSE is in pixel^2 space, we scale it down by a factor of 1000 to match IoU's magnitude
        mse_weight = 0.001 
        iou_weight = 1.0
        
        # Define a combined criterion function
        def criterion(outputs, targets):
            return (mse_weight * mse_loss(outputs, targets)) + (iou_weight * iou_loss(outputs, targets))
            
        save_name = "localizer.pth"
        
    elif args.task == "segmentation":
        model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)
        
        ce_loss = nn.CrossEntropyLoss()
        
        # Define a simple Dice Loss function
        def dice_loss(pred, target, smooth=1e-5):
            # pred: [B, C, H, W] logits
            # target: [B, H, W] class indices
            pred = torch.softmax(pred, dim=1)
            target_one_hot = torch.nn.functional.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
            
            intersection = (pred * target_one_hot).sum(dim=(2, 3))
            cardinality = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
            
            dice = (2. * intersection + smooth) / (cardinality + smooth)
            return 1 - dice.mean()
            
        def criterion(outputs, targets):
            # Combine them: CE helps stabilize early training, Dice pushes the final overlap
            return ce_loss(outputs, targets) + dice_loss(outputs, targets)
            
        save_name = "unet.pth"

    # Apply freezing strategy to the backbone
    apply_transfer_strategy(model, args.transfer_strategy)

    model = model.to(device)
    
    # CRITICAL: Filter parameters so the optimizer only updates unfrozen weights
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            
            # Target routing based on task
            if args.task == "classification":
                targets = batch["class_label"].to(device)
            elif args.task == "localization":
                targets = batch["bbox"].to(device)
            elif args.task == "segmentation":
                targets = batch["segmentation_mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / max(len(train_loader), 1)
        
        # TODO: Implement Validation Loop here

        # =========================
        # Validation Loop
        # =========================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)

                if args.task == "classification":
                    targets = batch["class_label"].to(device)

                elif args.task == "localization":
                    targets = batch["bbox"].to(device)

                elif args.task == "segmentation":
                    targets = batch["segmentation_mask"].to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        

    # Save Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_payload = {
        "state_dict": model.state_dict(),
        "epoch": args.epochs,
    }
    torch.save(checkpoint_payload, os.path.join("checkpoints", save_name))
    print(f"Saved {save_name} to checkpoints folder.")

if __name__ == "__main__":
    main()