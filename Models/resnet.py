import torchvision.models as models
import torch.nn as nn

# def get_model(num_classes=10):
#     model = models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model


class ResNet18Modified(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(ResNet18Modified, self).__init__()
        self.model = models.resnet18(pretrained=False)

        # If input channels are not 3 (e.g., grayscale images)
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the last layer for CIFAR-10 (10 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)