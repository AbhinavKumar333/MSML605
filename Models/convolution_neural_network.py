import torch
import torch.nn as nn
from torchvision import models

class VGG16Modified(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(VGG16Modified, self).__init__()

        # Load VGG-16 model
        self.model = models.vgg16(pretrained=False)

        # Modify the first conv layer to accept 1-channel input (e.g., for MNIST)
        if input_channels == 1:
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Modify the classifier to match the number of output classes
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)