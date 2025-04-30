import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_channels=0, pretrained=False):
        super().__init__()
        self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
        self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(8*8*128, 256), nn.ReLU(),
                        nn.Linear(256, num_classes)
                    )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
