import torch.nn as nn
from torchvision import models

def build_model(num_classes: int = 2) -> nn.Module:

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model