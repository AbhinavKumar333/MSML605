import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

class VGG16Modified(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, pretrained=True):
        super(VGG16Modified, self).__init__()

        # Load pretrained VGG-16 model with weights
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = models.vgg16(weights=weights)

        # Adjust input channels if needed (e.g., grayscale input)
        if input_channels != 3:
            self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)

        # Replace final classifier layer for CIFAR-10
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)