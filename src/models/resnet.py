import torch.nn as nn
from torchvision import models
import torch

def create_resnet50(num_classes: int = 7):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))
    model = model.to(device)
    print(f"✅ ResNet50 created: {sum(p.numel() for p in model.parameters())} total parameters")
    return model