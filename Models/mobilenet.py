import torchvision.models as models
import torch.nn as nn

# def get_model(num_classes=10):
#     model = models.mobilenet_v2(weights=None)
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
#     return model


class MobileNetV2Modified(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(MobileNetV2Modified, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)

        # Modify the first conv layer if input_channels != 3
        if input_channels != 3:
            self.model.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Modify the last fully connected layer
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)