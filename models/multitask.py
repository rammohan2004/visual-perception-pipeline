"""Unified multi-task model
"""

import torch
import torch.nn as nn
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
import os

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        import gdown
        gdown.download(id="<1Crs8ZYF2y4jqP2DzjyPdbSQMoEFe9xuE>", output=classifier_path, quiet=False)
        gdown.download(id="<1jFAGzqcahIn7iMgsCsam_1Of1nmA4gZT>", output=localizer_path, quiet=False)
        gdown.download(id="<1Tjj5ttTOM5lhMKFrW7pVlWMq_kvwNt9s>", output=unet_path, quiet=False)
        
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer = VGG11Localizer(in_channels=in_channels)
        self.unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        def load_weights(model, path):
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location="cpu")
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                print(f"Warning: Checkpoint {path} not found.")

        load_weights(self.classifier, classifier_path)
        load_weights(self.localizer, localizer_path)
        load_weights(self.unet, unet_path)

        self.shared_encoder = self.unet.encoder


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.

        #shared encoder
        bottleneck, features = self.shared_encoder(x, return_features=True)

        #flatten
        flattened_bottleneck = torch.flatten(bottleneck, start_dim=1)

        #classification
        class_out = self.classifier.head(flattened_bottleneck)

        #localization
        loc_out = self.localizer.head(flattened_bottleneck)

        #segmentation
        seg_out = bottleneck

        seg_out = self.unet.upconv5(seg_out)
        seg_out = torch.cat([seg_out, features["block5"]], dim=1)
        seg_out = self.unet.dec5(seg_out)

        #14 to 28
        seg_out = self.unet.upconv4(seg_out)
        seg_out = torch.cat([seg_out, features["block4"]], dim=1)
        seg_out = self.unet.dec4(seg_out)

        #28 to 56
        seg_out = self.unet.upconv3(seg_out)
        seg_out = torch.cat([seg_out, features["block3"]], dim=1)
        seg_out = self.unet.dec3(seg_out)

        #56 to 112
        seg_out = self.unet.upconv2(seg_out)
        seg_out = torch.cat([seg_out, features["block2"]], dim=1)
        seg_out = self.unet.dec2(seg_out)

        #112 to 224
        seg_out = self.unet.upconv1(seg_out)
        seg_out = torch.cat([seg_out, features["block1"]], dim=1)
        seg_out = self.unet.dec1(seg_out)

        #final segmentation output
        seg_out = self.unet.final_conv(seg_out)

        #return dict
        return {
            'classification': class_out,
            'localization': loc_out,
            'segmentation': seg_out
        }