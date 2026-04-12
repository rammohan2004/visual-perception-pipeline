"""Inference and evaluation for WandB Report"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import wandb
import urllib.request
from PIL import Image
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from train import get_transforms, dice_score

def compute_iou(pred_box, target_box):
    px, py, pw, ph = pred_box
    tx, ty, tw, th = target_box
    
    px1, py1, px2, py2 = px - pw/2, py - ph/2, px + pw/2, py + ph/2
    tx1, ty1, tx2, ty2 = tx - tw/2, ty - th/2, tx + tw/2, ty + th/2
    
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area = max(0, pw) * max(0, ph)
    target_area = max(0, tw) * max(0, th)
    union_area = pred_area + target_area - inter_area
    
    if union_area <= 0: return 0.0
    return inter_area / union_area

def task_2_4_feature_maps(device, val_loader):
    print("Running Task 2.4: Feature Maps (Table)...")
    model = VGG11Classifier(num_classes=37).to(device)
    model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device)["state_dict"])
    model.eval()

    batch = next(iter(val_loader))
    img = batch["image"][0].unsqueeze(0).to(device)

    with torch.no_grad():
        _, features = model.encoder(img, return_features=True)
    
    block1_map = features["block1"][0, 0].cpu().numpy() 
    block5_map = features["block5"][0, 0].cpu().numpy() 

    # Create W&B Table
    table = wandb.Table(columns=["Original Image", "Block 1 Feature Map", "Block 5 Feature Map"])
    
    display_img = img[0].permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    display_img = np.clip(display_img, 0, 1)

    fig1, ax1 = plt.subplots(); ax1.imshow(display_img); ax1.axis('off')
    fig2, ax2 = plt.subplots(); ax2.imshow(block1_map, cmap='viridis'); ax2.axis('off')
    fig3, ax3 = plt.subplots(); ax3.imshow(block5_map, cmap='viridis'); ax3.axis('off')
    
    table.add_data(wandb.Image(fig1), wandb.Image(fig2), wandb.Image(fig3))
    plt.close('all')

    wandb.log({"Task 2.4 - Feature Maps": table})

def task_2_5_object_detection(device, val_loader):
    print("Running Task 2.5: Object Detection (Table)...")
    model = VGG11Localizer().to(device)
    model.load_state_dict(torch.load("checkpoints/localizer.pth", map_location=device)["state_dict"])
    model.eval()
    
    class_model = VGG11Classifier().to(device)
    class_model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device)["state_dict"])
    class_model.eval()

    table = wandb.Table(columns=["Image", "Confidence Score", "IoU"])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 10: break
            img = batch["image"].to(device)
            target_box = batch["bbox"].to(device)
            
            pred_box = model(img)
            logits = class_model(img)
            confidence = F.softmax(logits, dim=1).max(dim=1)[0].item()
            
            iou = compute_iou(pred_box[0].cpu().numpy(), target_box[0].cpu().numpy())
            
            display_img = img[0].permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            display_img = np.clip(display_img, 0, 1)
            
            fig, ax = plt.subplots(1)
            ax.imshow(display_img)
            
            tx, ty, tw, th=target_box[0].cpu().numpy()
            ax.add_patch(patches.Rectangle((tx - tw/2, ty - th/2), tw, th, linewidth=2, edgecolor='g', facecolor='none', label='GT'))
            px, py, pw, ph = pred_box[0].cpu().numpy()
            ax.add_patch(patches.Rectangle((px - pw/2, py - ph/2), pw, ph, linewidth=2, edgecolor='r', facecolor='none', label='Pred'))
            ax.legend(); ax.axis('off')
            
            table.add_data(wandb.Image(fig), round(confidence, 4), round(iou, 4))
            plt.close(fig)

    wandb.log({"Task 2.5 - Object Detection (Confidence & IoU)": table})

def task_2_6_segmentation(device, val_loader):
    print("Running Task 2.6: Segmentation (Table)...")
    model = VGG11UNet(num_classes=3).to(device)
    model.load_state_dict(torch.load("checkpoints/unet.pth", map_location=device)["state_dict"])
    model.eval()

    table = wandb.Table(columns=["Original Image", "Ground Truth Mask", "Predicted Mask"])

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5: break
            img = batch["image"].to(device)
            target_mask = batch["segmentation_mask"].to(device)
            
            pred_logits = model(img)
            pred_mask = torch.argmax(pred_logits, dim=1)
            
            display_img = img[0].permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            display_img = np.clip(display_img, 0, 1)
            
            fig1,ax1 =plt.subplots(); ax1.imshow(display_img); ax1.axis('off')
            fig2, ax2 = plt.subplots(); ax2.imshow(target_mask[0].cpu().numpy(), vmin=0, vmax=2); ax2.axis('off')
            fig3, ax3 = plt.subplots(); ax3.imshow(pred_mask[0].cpu().numpy(), vmin=0, vmax=2); ax3.axis('off')
            
            table.add_data(wandb.Image(fig1),wandb.Image(fig2), wandb.Image(fig3))
            plt.close('all')

    wandb.log({"Task 2.6 - Segmentation Samples": table})

def task_2_7_wild_images(device):
    print("Running Task 2.7: In-the-wild Images (Table)...")
    model =MultiTaskPerceptionModel(classifier_path="checkpoints/classifier.pth", 
                                     localizer_path="checkpoints/localizer.pth", 
                                     unet_path="checkpoints/unet.pth").to(device)
    model.eval()

    urls =[
        "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg", # Safe Dog URL
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    ]
    
    transform= A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    table = wandb.Table(columns=["Input URL", "Detection (Class & BBox)", "Segmentation Mask"])

    for i, url in enumerate(urls):
        req =urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        img_arr= np.asarray(bytearray(response.read()), dtype=np.uint8)
        
        import cv2
        img_raw = cv2.imdecode(img_arr, -1)
        img_raw =cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=img_raw)
        img_tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
        
        pred_box =outputs['localization'][0].cpu().numpy()
        pred_mask = torch.argmax(outputs['segmentation'], dim=1)[0].cpu().numpy()
        pred_class= torch.argmax(outputs['classification'], dim=1).item()
        
        display_img =img_tensor[0].permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        display_img= np.clip(display_img, 0, 1)

        # Draw Detection Figure
        fig1, ax1 = plt.subplots()
        ax1.imshow(display_img)
        ax1.set_title(f"Class ID: {pred_class}")
        px, py, pw, ph= pred_box
        ax1.add_patch(patches.Rectangle((px - pw/2, py - ph/2), pw, ph, linewidth=2, edgecolor='r', facecolor='none'))
        ax1.axis('off')
        
        # Draw Segmentation Figure
        fig2, ax2 =plt.subplots()
        ax2.imshow(pred_mask,vmin=0, vmax=2)
        ax2.axis('off')
        
        table.add_data(url,wandb.Image(fig1), wandb.Image(fig2))
        plt.close('all')

    wandb.log({"Task 2.7 -In-the-Wild Pipeline": table})

if __name__ == "__main__":
    wandb.init(project="da6401_assignment_2",name="inference_and_visualizations")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, val_transform =get_transforms()
    # Uses train split to bypass missing test annotations issue
    dataset =OxfordIIITPetDataset(root_dir="/kaggle/input/datasets/julinmaloof/the-oxfordiiit-pet-dataset", split="train", transforms=val_transform)
    val_loader =DataLoader(dataset, batch_size=1, shuffle=False)

    task_2_4_feature_maps(device, val_loader)
    task_2_5_object_detection(device, val_loader)
    task_2_6_segmentation(device, val_loader)
    task_2_7_wild_images(device)
    
    wandb.finish()
    print("All inference tasks logged to Weights & Biases successfully!")