"""Training entrypoint
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import wandb
import numpy as np
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


'''Exact Metrics & Losses'''
def dice_loss(pred_logits, target, num_classes=3, eps=1e-6):
    """
    pred_logits: [B, num_classes, H, W] raw logits
    target:      [B, H, W] integer class indices {0,1,2}
    returns:     scalar loss value in [0, 1]
    """
    B, C, H, W = pred_logits.shape
    
    pred_prob = torch.softmax(pred_logits, dim=1)
    
    # one-hot encoding target to [B, C, H, W]
    one_hot=torch.zeros(B, num_classes, H, W, device=target.device)
    one_hot.scatter_(1, target.unsqueeze(1).long(), 1)
    
    # computing per-class Dice score
    probs=pred_prob.view(B, C, -1)
    one_hot = one_hot.view(B, C, -1)

    intersection =torch.sum(probs * one_hot, dim=2)
    pred_sum = torch.sum(probs, dim=2)
    target_sum = torch.sum(one_hot, dim=2)

    dice_sc = (2.*intersection+eps) / (pred_sum+ target_sum + eps)
    return 1 - dice_sc.mean()

def dice_score(pred_logits, target, num_classes=3, eps=1e-6):
    """
    pred_logits: [B, num_classes, H, W] raw logits
    target:      [B, H, W] integer class indices {0,1,2}
    returns:     scalar score value
    """
    B, C, H, W = pred_logits.shape
    
    pred_prob = torch.softmax(pred_logits, dim=1)
    
    one_hot = torch.zeros(B, num_classes, H, W, device=target.device)
    one_hot.scatter_(1, target.unsqueeze(1).long(), 1)
    
    probs = pred_prob.view(B, C, -1)
    one_hot = one_hot.view(B, C, -1)

    intersection = torch.sum(probs * one_hot, dim=2)
    pred_sum = torch.sum(probs, dim=2)
    target_sum = torch.sum(one_hot, dim=2)

    dice_sc = (2. * intersection + eps) / (pred_sum + target_sum + eps)
    return dice_sc.mean()

'''Transforms & Freezing Strategies'''

def get_transforms():
    """Returns train and validation albumentations transforms."""
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    return train_transform, val_transform

def apply_transfer_strategy(model, strategy):
    """Applies the freezing logic to the model's encoder."""
    if strategy == "strict":
        print("[Transfer Learning] Strict: Freezing entire encoder.")
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == "partial":
        print("[Transfer Learning] Partial: Freezing blocks 1, 2, and 3.")
        for param in model.encoder.block1.parameters(): param.requires_grad = False
        for param in model.encoder.block2.parameters(): param.requires_grad = False
        for param in model.encoder.block3.parameters(): param.requires_grad = False
    elif strategy == "full":
        print("[Transfer Learning] Full: Updating all encoder weights end-to-end.")
        pass 



#Main Execution

def main():
    parser = argparse.ArgumentParser(description="Train DA6401 Assignment 2 Models")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "localization", "segmentation"])
    parser.add_argument("--transfer_strategy", type=str, default="strict", choices=["strict", "partial", "full"])
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

    '''Dataset + Loader''' 
    
    full_dataset = OxfordIIITPetDataset(
        root_dir=args.data_dir,
        split="train",
        transforms=train_transform
    )

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    #Generate exact indices
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = Subset(full_dataset, train_indices)
    
    val_full_dataset = OxfordIIITPetDataset(
        root_dir=args.data_dir,
        split="train",
        transforms=val_transform
    )
    val_data = Subset(val_full_dataset, val_indices)

    print(f"Total: {len(full_dataset)} | Train: {len(train_data)}, Val: {len(val_data)}")

    use_pin_memory = torch.cuda.is_available()
    optimal_workers = min(4, os.cpu_count())

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=optimal_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=optimal_workers, pin_memory=use_pin_memory)
    
    '''Model & Loss Init'''

    if args.task == "classification":
        model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)
        criterion = nn.CrossEntropyLoss()
        save_name = "classifier.pth"
        
    elif args.task == "localization":
        model = VGG11Localizer(dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)
        mse_loss = nn.MSELoss()
        iou_loss_mean = IoULoss(reduction="mean")
        iou_loss_none = IoULoss(reduction="none") 
        
        #unweighted sum loss
        def criterion(outputs, targets):
            return mse_loss(outputs, targets) + iou_loss_mean(outputs, targets)
            
        save_name = "localizer.pth"
        
    elif args.task == "segmentation":
        model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p, use_batchnorm=args.use_batchnorm)
        ce_loss = nn.CrossEntropyLoss()
        
        #combined loss
        def criterion(outputs, targets):
            return ce_loss(outputs, targets.long()) + dice_loss(outputs, targets)
            
        save_name = "unet.pth"

    model = model.to(device)

    '''Transfer Weights & 0.1x LR logic'''

    use_differential_lr = False

    if args.task in ["localization", "segmentation"]:
        if os.path.exists("checkpoints/classifier.pth"):
            print("=>Loading pre-trained encoder weights from classifier.pth")
            checkpoint = torch.load("checkpoints/classifier.pth", map_location=device)
            state = checkpoint.get("state_dict", checkpoint)
            encoder_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
            model.encoder.load_state_dict(encoder_state, strict=True)
            use_differential_lr = True
        else:
            print("=> No classifier checkpoint found, training encoder from scratch.")

        apply_transfer_strategy(model, args.transfer_strategy)

    ''' optimizer & scheduler setup '''

    if use_differential_lr and args.transfer_strategy in ["partial", "full"]:
        print("=> Applying 0.1x Learning Rate scaling to unfrozen encoder parameters.")
        encoder_params =filter(lambda p: p.requires_grad, model.encoder.parameters())
        head_params =[p for n, p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
        
        optimizer =optim.Adam([
            {"params": encoder_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr}
        ])
    else:
        trainable_params= filter(lambda p: p.requires_grad, model.parameters())
        optimizer =optim.Adam(trainable_params, lr=args.lr)
    
    scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    best_val_metric = 0.0

    '''Training Loop'''
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_metric_accum = 0.0
        all_train_preds,all_train_labels = [], []
        
        #Dictionary unpacking
        for batch in train_loader:
            images = batch["image"].to(device)
            
            #Target routing with strict clamping for masks
            if args.task == "classification":
                targets =batch["class_label"].to(device)
            elif args.task == "localization":
                targets = batch["bbox"].to(device).float()
            elif args.task =="segmentation":
                targets = torch.clamp(batch["segmentation_mask"].to(device).long(), min=0, max=2)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss +=loss.item()
            
            #Metric accumulation
            with torch.no_grad():
                if args.task =="classification":
                    preds =torch.argmax(outputs, dim=1)
                    all_train_preds.extend(preds.cpu().numpy())
                    all_train_labels.extend(targets.cpu().numpy())
                elif args.task == "localization":
                    per_sample_iou = 1.0 - iou_loss_none(outputs.detach(), targets) 
                    train_metric_accum += per_sample_iou.mean().item()
                elif args.task =="segmentation":
                    train_metric_accum += dice_score(outputs.detach(), targets).item()
            
        avg_train_loss = running_loss / max(len(train_loader), 1)
        
        if args.task =="classification":
            train_metric = f1_score(all_train_labels, all_train_preds, average='macro')
        else:
            train_metric = train_metric_accum / max(len(train_loader), 1)

        '''Validation loop'''
        model.eval()
        val_loss = 0.0
        val_metric_accum = 0.0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            #Dictionary unpacking
            for batch in val_loader:
                images =batch["image"].to(device)

                #Target routing with strict clamping for masks
                if args.task =="classification":
                    targets =batch["class_label"].to(device)
                elif args.task== "localization":
                    targets =batch["bbox"].to(device).float()
                elif args.task== "segmentation":
                    targets = torch.clamp(batch["segmentation_mask"].to(device).long(), min=0, max=2)

                outputs =model(images)
                loss= criterion(outputs, targets)
                val_loss +=loss.item()

                if args.task == "classification":
                    preds =torch.argmax(outputs, dim=1)
                    all_val_preds.extend(preds.cpu().numpy())
                    all_val_labels.extend(targets.cpu().numpy())
                elif args.task == "localization":
                    per_sample_iou = 1.0 - iou_loss_none(outputs, targets) 
                    val_metric_accum +=per_sample_iou.mean().item()
                elif args.task == "segmentation":
                    val_metric_accum+= dice_score(outputs, targets).item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        if args.task == "classification":
            val_metric =f1_score(all_val_labels, all_val_preds, average='macro')
        else:
            val_metric = val_metric_accum / max(len(val_loader), 1)

        scheduler.step(val_metric)

        metric_name ="F1" if args.task =="classification" else ("IoU" if args.task == "localization" else "Dice")
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Train {metric_name}: {train_metric:.4f}, Val {metric_name}: {val_metric:.4f}")

        #Checkpoint saving logic
        if val_metric >best_val_metric:
            best_val_metric = val_metric
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_payload = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_val_metric
            }
            torch.save(checkpoint_payload, os.path.join("checkpoints", save_name))
            print(f"--> Saved best {save_name} (Val {metric_name}: {best_val_metric:.4f})")

        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            f"train/{metric_name.lower()}": train_metric,
            "val/loss": avg_val_loss,
            f"val/{metric_name.lower()}": val_metric,
            f"best_val/{metric_name.lower()}": best_val_metric
        })

    wandb.finish()

if __name__ == "__main__":
    main()