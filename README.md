## Wandb Report Link
You can find the Report on [WandB Report](
https://wandb.ai/cs25m017-indian-institute-of-technology-madras/da6401_assignment_2/reports/Untitled-Report--VmlldzoxNjQ1ODI0OA?accessToken=kus07g5zxc5ohe1fhxnq3sgmk563pumcfdxworahbxadymv856p71f6pompinoik)

## GitHub Repository
You can find the complete project on [GitHub](https://github.com/rammohan2004/visual-perception-pipeline).





# Complete Visual Perception Pipeline

This repository contains a unified deep learning pipeline built with PyTorch that performs simultaneous image classification, bounding box localization, and semantic segmentation. It was developed and trained using the Oxford-IIIT Pet Dataset.

## Project Overview

The goal of this project is to explore multi-task learning by sharing a single convolutional backbone (VGG11) across three distinct computer vision tasks. The project evaluates the architectural impact of Batch Normalization, Custom Dropout placement, and transfer learning strategies (freezing vs. fine-tuning) to mitigate task interference.

### Core Tasks
1. **Classification:** Identifies the specific breed of the pet (37 classes) using a linear classification head and Cross-Entropy Loss.
2. **Localization:** Predicts the bounding box around the animal's head using a regression head trained with a combination of Mean Squared Error (MSE) and a custom Intersection over Union (IoU) Loss.
3. **Segmentation:** Generates a pixel-wise trimap mask (foreground, background, boundary) using a U-Net style decoder that utilizes skip connections from the shared VGG11 encoder, trained with a combined Cross-Entropy and Dice Loss.

## Dataset

The model utilizes the **Oxford-IIIT Pet Dataset**, which contains:
- 37 category pet images with variations in scale, pose, and lighting.
- Ground truth bounding box annotations (XML format).
- Trimap segmentation masks (PNG format).

Data augmentation is handled via the albumentations library, applying random horizontal flips, color jitter, and slight rotations during the training phase.

## Repository Structure

    ├── data/
    │   └── pets_dataset.py       # Custom PyTorch Dataset loader and XML parser
    ├── losses/
    │   └── iou_loss.py           # Custom IoU loss implementation for bounding boxes
    ├── models/
    │   ├── layers.py             # Custom Dropout implementations
    │   ├── vgg11.py              # Shared VGG11 Encoder with conditional Batch Normalization
    │   ├── classification.py     # Classification head
    │   ├── localization.py       # Bounding box regression head
    │   ├── segmentation.py       # U-Net style decoder
    │   └── multitask.py          # Unified multi-task inference pipeline
    ├── train.py                  # Main training entrypoint with W&B logging
    └── README.md


## Requirements

- Python 3.8+
- PyTorch
- torchvision
- albumentations
- Weights & Biases (wandb)
- scikit-learn
- Pillow
- gdown

## Usage

### Training Isolated Models

You can train individual task models from scratch or using specific transfer learning strategies via the command line interface in train.py.

**1. Train the Classifier:**
    python train.py --task classification --epochs 20 --batch_size 32 --lr 1e-4 --use_batchnorm

**2. Train the Localizer (Strict Freeze Strategy):**
    python train.py --task localization --transfer_strategy strict --use_batchnorm

**3. Train the Segmentation U-Net:**
    python train.py --task segmentation --transfer_strategy strict --use_batchnorm


### Multi-Task Inference
Once the individual weights (classifier.pth, localizer.pth, unet.pth) are generated, the MultiTaskPerceptionModel in multitask.py combines them. It forces all three task heads to utilize the frozen classification backbone to output simultaneous predictions.

## Key Architectural Insights

Through comprehensive W&B logging and experimentation, several key conclusions were drawn regarding multi-task architectures:

- **Mitigating Task Interference:** Utilizing a "Strict Freeze" strategy for the shared backbone proved highly effective. When attempting "Full Fine-Tuning", the tasks experienced severe interference. The localization and segmentation heads altered the shared weights, destroying the original classification features. Freezing the classification backbone forced all tasks to adapt to a stable, shared representation.
- **Batch Normalization:** Implementing Batch Normalization inside the shared encoder was critical. It successfully mitigated internal covariate shift, allowing the model to utilize a higher maximum stable learning rate without dying (outputting pure zeros) during early epochs.
- **Loss Formulation for Class Imbalance:** Relying solely on Cross-Entropy loss for segmentation resulted in artificially high accuracy due to background pixel dominance. Implementing a custom Dice Loss was essential to force the network to accurately learn the specific spatial boundaries of the foreground object.

## Acknowledgements

Developed as part of the DA6401 coursework. Logging and metric tracking powered by Weights & Biases.