import torch.nn as nn
from torchvision import models


class MyAgeClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # ResNet-18 backbone (must be trained from scratch)
        self.backbone = models.resnet18(weights=None)

        num_ftrs = self.backbone.fc.in_features

        # Modified classification head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)