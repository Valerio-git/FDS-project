import torch.nn as nn
from torchvision import models
import torch

def create_resnet50(num_classes: int = 7):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    ResNet50 pre-addestrata ImageNet-1K (1.28M immagini, 1000 classi)
    Sostituisce FC finale: 2048 → num_classes (7 materiali rifiuti)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    print(f"✅ ResNet50 creata: {sum(p.numel() for p in model.parameters())} parametri totali")
    return model

def setup_feature_extraction(model: nn.Module):
    """
    FASE 1: Feature Extraction [1]
    - Congela layer1-4 (feature extractor ImageNet)
    - Allena SOLO FC finale (adattamento alle 7 classi rifiuti)
    LR alto (1e-3) perché solo ~1% parametri allenabili
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def setup_fine_tuning_last_block(model: nn.Module):
    """
    FASE 2: Fine Tuning parziale [2]
    - Sblocca layer4 (caratteristiche high-level) + FC
    - Congela layer1-3 (caratteristiche low-level ImageNet)
    LR basso (1e-4): fine adattamento senza distruggere pesi pre-addestrati
    """
    for name, param in model.named_parameters():
        # Sblocca ULTIMO blocco conv (layer4) + FC
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

def get_trainable_parameters(model):
    """
    Tramite model.named_parameters ho estratto tutti i parametri, ognuno ha un attributo
    param.requires_grad che se = True permette a pyhton di calcolare il gradiente.
    Qui filtriamo solamente i parametri = True (quindi no quelli dei layer congelati)
    """
    return filter(lambda p: p.requires_grad, model.parameters())