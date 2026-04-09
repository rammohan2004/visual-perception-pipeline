"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, use_batchnorm: bool = True):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        # encoder
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batchnorm=use_batchnorm)
        
        # Helper for conditional batch norm
        def bn(num_features):
            return nn.BatchNorm2d(num_features) if use_batchnorm else nn.Identity()
        
        # upconv (Channels are kept wide here to preserve abstract features)
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # dec5: 512 (up) + 512 (skip) = 1024 → 512
        self.dec5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            bn(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            bn(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p)
        )

        # dec4: 512 (up) + 512 (skip) = 1024 → 256
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            bn(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            bn(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p)
        )

        # dec3: 256 (up) + 256 (skip) = 512 → 128
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            bn(128),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            bn(128),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p)
        )

        # dec2: 128 (up) + 128 (skip) = 256 → 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            bn(64),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p)
        )

        # dec1: 64 (up) + 64 (skip) = 128 → 32
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            bn(32),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p)
        )

        # final layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # encoder
        bottleneck, features = self.encoder(x, return_features=True)

        # decoder path
        # 7 to 14
        x = self.upconv5(bottleneck)
        x = torch.cat([x, features["block5"]], dim=1)
        x = self.dec5(x)

        # 14 to 28
        x = self.upconv4(x)
        x = torch.cat([x, features["block4"]], dim=1)
        x = self.dec4(x)

        # 28 to 56
        x = self.upconv3(x)
        x = torch.cat([x, features["block3"]], dim=1)
        x = self.dec3(x)

        # 56 to 112
        x = self.upconv2(x)
        x = torch.cat([x, features["block2"]], dim=1)
        x = self.dec2(x)

        # 112 to 224
        x = self.upconv1(x)
        x = torch.cat([x, features["block1"]], dim=1)
        x = self.dec1(x)

        # final output
        x = self.final_conv(x)

        return x