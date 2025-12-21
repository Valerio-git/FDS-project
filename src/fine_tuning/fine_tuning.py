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

    To use as first step of fine-tuning.
    """
    for name, param in model.named_parameters():
        if name.startswith(("fc1", "fc2", "fc3")):
            param.requires_grad = True
        else:
            param.requires_grad = False
    '''model.named_parameters() contains all the parameters of the model, each one is either
    a conv layer or a fully connected one, so they are called: conv1.weight, conv2.bias,...,
    fc1.weight, fc2.bias...
    then it calculates the gradient (so trains) only fc1 and fc2'''


def unfreeze_last_conv_block(model: CNN) -> None:
    """
    STEP 2 fine-tuning:
    - make conv3, bn3, fc1 and fc2 trainable
    - conv1/bn1/conv2/bn2 remain frozen (more generic features)

    To use after STEP 1, for a deeper adaptation.
    """
    for name, param in model.named_parameters():
        if name.startswith(("conv3", "bn3", "fc1", "fc2", "fc3")):
            param.requires_grad = True
        # qui congela solamente i primi due layer convoluzionali

def setup_feature_extraction(model: nn.Module):
    """
    FASE 1: Feature Extraction [1]
    - Freeze layer1-4 (feature extractor ImageNet)
    - Train ONLY FC final (adaptation to 7 waste classes)
    LR high (1e-3) because only ~1% parameters are trainable
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def setup_fine_tuning_last_block(model: nn.Module):
    """
    FASE 2: Partial Fine Tuning
    - Unfreeze layer4 (high-level features) + FC
    - Freeze layer1-3 (low-level ImageNet features)
    LR low (1e-4): fine adaptation without destroying pre-trained weights
    """
    for name, param in model.named_parameters():
        # Sblocca ULTIMO blocco conv (layer4) + FC
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
