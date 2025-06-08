
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

class DualResNet(nn.Module):
    def __init__(self, num_classes=1652, pretrained=True):
        super(DualResNet, self).__init__()
        self.backbone1 = models.resnet18(pretrained=pretrained)
        self.backbone2 = models.resnet18(pretrained=pretrained)
        self.backbone1.fc = nn.Identity()
        self.backbone2.fc = nn.Identity()
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)

    def forward(self, x1, x2):
        f1 = self.backbone1(x1)
        f2 = self.backbone2(x2)
        out1 = self.classifier1(f1)
        out2 = self.classifier2(f2)
        return out1, out2, f1, f2

def get_num_classes(data_dir):
      sat_dir = Path(data_dir) / "satellite"
      class_folders = [f for f in sat_dir.iterdir() if f.is_dir()]
      return len(class_folders)
