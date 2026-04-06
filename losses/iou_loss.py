"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
        # TODO: validate reduction in {"none", "mean", "sum"}.
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}")
        

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.

        #extracting cx cy w h from pred_boxes
        pred_cx = pred_boxes[:, 0]
        pred_cy = pred_boxes[:, 1]
        pred_w  = pred_boxes[:, 2]
        pred_h  = pred_boxes[:, 3]

        #extracting cx cy w h from target_boxes
        target_cx = target_boxes[:, 0]
        target_cy =target_boxes[:, 1]
        target_w  =target_boxes[:,2]
        target_h  = target_boxes[:,3]

        #converting pred_boxes to (xmin, ymin, xmax, ymax)
        pred_xmin =pred_cx -pred_w / 2
        pred_ymin = pred_cy -pred_h / 2
        pred_xmax = pred_cx + pred_w/ 2
        pred_ymax =pred_cy + pred_h / 2

        #converting target_boxes to (xmin, ymin, xmax, ymax)
        target_xmin = target_cx - target_w/2
        target_ymin = target_cy - target_h/2
        target_xmax =target_cx + target_w/ 2
        target_ymax = target_cy + target_h / 2

        #intersection rectangle
        inter_xmin =torch.max(pred_xmin, target_xmin)
        inter_ymin = torch.max(pred_ymin,target_ymin)
        inter_xmax = torch.min(pred_xmax,target_xmax)
        inter_ymax= torch.min(pred_ymax,target_ymax)

        #intersection area
        inter_w = torch.clamp(inter_xmax-inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax-inter_ymin, min=0)
        inter_area = inter_w *inter_h

        pred_w =torch.clamp(pred_w, min=0)
        pred_h =torch.clamp(pred_h, min=0)
        target_w = torch.clamp(target_w, min=0)
        target_h =torch.clamp(target_h, min=0)

        #area of both boxes
        pred_area = pred_w *pred_h
        target_area = target_w *target_h

        #union
        union_area = pred_area +target_area - inter_area

        #IoU
        iou = inter_area /(union_area + self.eps)

        #loss = 1 - IoU
        loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss