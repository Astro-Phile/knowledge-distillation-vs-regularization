# --- BLOCK: MODEL DEFINITION  ---
import torch
import torch.nn as nn
from torchvision import models

class MyAgeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)

        num_ftrs = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if self.training:
            return self.backbone(x)
        else:
            out_original = self.backbone(x)
            out_flipped = self.backbone(torch.flip(x, dims=[3]))
            return (out_original + out_flipped) / 2.0
