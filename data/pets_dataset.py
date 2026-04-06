"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, root_dir: str, split: str = "train", transforms=None):
        """
        Args:
            root_dir: Path to the 'dataset' folder (containing 'images' and 'annotations').
            split: 'train' or 'test'.
            transforms: Albumentations transforms to apply.
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")
        self.xmls_dir = os.path.join(root_dir, "annotations", "xmls")
        
        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root_dir, "annotations", split_file)
        
        self.samples = []
        
        with open(split_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                filename = parts[0]
                class_id = int(parts[1]) - 1 
                
                img_path = os.path.join(self.images_dir, f"{filename}.jpg")
                mask_path = os.path.join(self.masks_dir, f"{filename}.png")
                xml_path = os.path.join(self.xmls_dir, f"{filename}.xml")
                
                if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(xml_path):
                    self.samples.append({
                        "filename": filename,
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "xml_path": xml_path,
                        "class_id": class_id
                    })
                    
        print(f"Loaded {len(self.samples)} valid samples for {split} split.")

    def _parse_xml_bbox(self, xml_path: str):

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        return [xmin, ymin, xmax, ymax]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pil_image = Image.open(sample["img_path"]).convert("RGB")
        image = np.array(pil_image)
        pil_mask = Image.open(sample["mask_path"])
        mask = np.array(pil_mask, dtype=np.int64)
        mask = mask - 1 
        bbox = self._parse_xml_bbox(sample["xml_path"])
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask, bboxes=[bbox], class_labels=[sample["class_id"]])
            image = transformed['image']
            mask = transformed['mask']
            bbox = transformed['bboxes'][0] 
    
        xmin, ymin, xmax, ymax = bbox
        x_center =(xmin + xmax) / 2.0
        y_center =(ymin + ymax) /2.0
        width =xmax - xmin
        height =ymax - ymin
        formatted_bbox = [x_center,y_center, width, height]
        
        return {
            "image": image, #albumentations transforms includes ToTensorV2
            "class_label": torch.tensor(sample["class_id"], dtype=torch.long),
            "bbox": torch.tensor(formatted_bbox, dtype=torch.float32),
            "segmentation_mask": torch.tensor(mask, dtype=torch.long)
        }