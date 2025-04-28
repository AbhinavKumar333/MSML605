import torchvision.models as models
import torch.nn as nn

def get_model(num_classes=10):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
