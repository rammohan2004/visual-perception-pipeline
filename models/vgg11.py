"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, use_batchnorm: bool = True):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        self.use_batchnorm = use_batchnorm
        def bn(num_features):
            return nn.BatchNorm2d(num_features) if self.use_batchnorm else nn.Identity()
        
        #block1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            bn(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            bn(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            bn(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            bn(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            bn(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            bn(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block5
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            bn(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            bn(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.
        features ={}

        #block 1
        x = self.block1(x)
        features["block1"]= x
        x =self.pool1(x)

        #block 2
        x =self.block2(x)
        features["block2"]= x
        x =self.pool2(x)

        #block 3
        x = self.block3(x)
        features["block3"] =x
        x = self.pool3(x)

        #block 4
        x = self.block4(x)
        features["block4"]=x
        x = self.pool4(x)

        #block 5
        x = self.block5(x)
        features["block5"]=x
        x = self.pool5(x)

        if return_features:
            return x, features
        return x
        
