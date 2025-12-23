from src.models.CNN import CNN
import torch
import torch.nn as nn
from torchvision import models

def load_cnn_from_checkpoint(checkpoint_path: str, num_classes: int, 
                             device: torch.device | None = None, strict: bool = True) -> CNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes = num_classes)
    state_dict = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state_dict["model_state_dict"], strict = strict)

    if device is not None:
        model.to(device)

    return model


def freeze_all_except_classifier(model: CNN) -> None:
    """
    STEP 1 fine-tuning:
    - freeze all convolutional layers
    - train only the classifier (fc1 and fc2)
    """
    for name, param in model.named_parameters():
        if name.startswith(("fc1", "fc2", "fc3")):
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_last_conv_block(model: CNN) -> None:
    """
    STEP 2 fine-tuning:
    - make conv3, bn3, fc1 and fc2 trainable (unfreez the 3rd conv layer)
    - conv1/bn1/conv2/bn2 remain frozen (more generic features)
    """
    for name, param in model.named_parameters():
        if name.startswith(("conv3", "bn3", "fc1", "fc2", "fc3")):
            param.requires_grad = True

def setup_feature_extraction(model: nn.Module):
    """
    STEP 1: Feature Extraction
    - Freeze layer1-4 (feature extractor ImageNet)
    - Train ONLY FC final (adaptation to 7 waste classes)
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def setup_fine_tuning_last_block(model: nn.Module):
    """
    STEP 2: Partial Fine Tuning
    - Unfreeze layer4 (high-level features) + FC
    - Freeze layer1-3 (low-level ImageNet features)
    """
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_trainable_parameters(model):
    """
    Through model.named_parameters I extracted all the parameters, each one has an attribute
    param.requires_grad which if = True allows python to compute the gradient.
    Here we filter only the parameters = True (so not those of the frozen layers)
    """
    return filter(lambda p: p.requires_grad, model.parameters())
