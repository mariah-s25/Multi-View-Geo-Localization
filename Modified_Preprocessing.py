
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path  

class PairedU1652Dataset(Dataset):
    def __init__(self, root_dir, transform_sat=None, transform_drone=None):
        self.root_dir = root_dir
  
        self.sat_dir = os.path.join(root_dir, 'satellite')
        self.drone_dir = os.path.join(root_dir, 'drone')
        
        self.transform_sat = transform_sat
        self.transform_drone = transform_drone
        
        self.sat_classes = sorted(os.listdir(self.sat_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.sat_classes)}
        
        self.sat_images = []
        self.drone_images = []
        self.labels = []
        
        for cls_name in self.sat_classes:
            sat_cls_path = os.path.join(self.sat_dir, cls_name)
            drone_cls_path = os.path.join(self.drone_dir, cls_name)
            
            sat_imgs = sorted(os.listdir(sat_cls_path))
            drone_imgs = sorted(os.listdir(drone_cls_path))
            
            assert len(sat_imgs) == 1, f"Expected exactly 1 satellite image per class '{cls_name}', but got {len(sat_imgs)}"
            
            sat_img_name = sat_imgs[0]
            sat_img_path = os.path.join(sat_cls_path, sat_img_name)
            
            for drone_img_name in drone_imgs:
                drone_img_path = os.path.join(drone_cls_path, drone_img_name)
                
                self.sat_images.append(sat_img_path)       # Repeat the same satellite image path
                self.drone_images.append(drone_img_path)
                self.labels.append(self.class_to_idx[cls_name])
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sat_img_path = self.sat_images[idx]
        drone_img_path = self.drone_images[idx]
        label = self.labels[idx]
        
        sat_img = Image.open(sat_img_path).convert('RGB')
        drone_img = Image.open(drone_img_path).convert('RGB')
        
        if self.transform_sat:
            sat_img = self.transform_sat(sat_img)
        if self.transform_drone:
            drone_img = self.transform_drone(drone_img)
        
        return sat_img, drone_img, label

