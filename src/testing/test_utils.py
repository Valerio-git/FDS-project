import os
import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import (transforms, models)
from typing import (Literal,Tuple)

from src.data_utils import (get_raw_dataset_path,
                            get_white_dataset_path)

from src.data.data_loader import WasteDataset
from src.models.CNN import CNN

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModelType = Literal["cnn", "resnet"]

def load_test_dataset(transform: transforms.Compose, white:bool = False) -> Tuple[WasteDataset, DataLoader]:
    if white:
        dataset_path = get_white_dataset_path()
    else:
        dataset_path = get_raw_dataset_path()
    test_dataset = WasteDataset(dataset_path, split="test", transform=transform, white = white)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_dataset, test_loader

def evaluate_test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
           
            all_preds.extend(preds.cpu().numpy()) # takes preds tensor and pass them to cpu, extend converts them to lists
            all_labels.extend(labels.cpu().numpy()) # same process for labels


    all_preds = np.array(all_preds) # coverting them into numpy arrays 
    all_labels = np.array(all_labels)

    accuracy = correct/total
    f1 = f1_score(all_labels, all_preds, average="macro")

    return accuracy, all_preds, all_labels, f1


def get_best_model_path(white: bool = False, model_type: ModelType = "cnn") -> str:
    current_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    if model_type == "resnet":
        model_path = os.path.join(base_dir, "src/checkpoints/resnet_stage2.pth")
    elif model_type == "cnn":
        if white:
            model_path = os.path.join(base_dir, "src/checkpoints/cnn_stage2.pth")
        else:
            model_path = os.path.join(base_dir, "src/checkpoints/cnn_stage1_A.pth")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return str(model_path)


def load_trained_model(num_classes: int, device: torch.device, white:bool = False, model_type: ModelType = "cnn") -> nn.Module:
    if model_type == "cnn":
        model = CNN(num_classes)
    elif model_type == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features  # 2048
        model.fc = nn.Linear(in_features, num_classes)
        model = model.to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    model_path = get_best_model_path(white=white, model_type=model_type)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    if white:
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    return model


def select_random_samples(dataset: WasteDataset, model: nn.Module, device: torch.device,
                           num_samples: int = 6,
                           ensure_different_labels: bool = True)-> tuple[list[torch.Tensor], list[int], list[int]]:
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    selected_images = []
    selected_labels = []
    selected_predicted = []

    with torch.no_grad():
        for idx in indices:
            
            image, label = dataset[idx]
            image = image.to(device).unsqueeze(0)
            
            output = model(image)
            _, predicted = torch.max(output, dim=1)
            predicted_label = predicted.item()

            if ensure_different_labels and label in selected_labels:
                continue

            selected_images.append(image.squeeze(0)) 
            selected_labels.append(label)
            selected_predicted.append(predicted_label)

            if len(selected_images) == num_samples:
                break
            
    return selected_images, selected_labels, selected_predicted


def denormalize_images(images: list[torch.Tensor], mean: list[float] = IMAGENET_MEAN, 
                       std: list[float] = IMAGENET_STD) -> list[torch.Tensor]:
    
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    denorm_images = []

    for img in images:
        img_denorm = img.cpu() * std_tensor + mean_tensor
        img_denorm = img_denorm.clamp(0, 1) 
        denorm_images.append(img_denorm)
    
    return denorm_images

def plot_predictions(images: list[torch.Tensor], true_labels: list[int], 
                     predicted_labels: list[int], class_names: list[str], grid_rows: int = 3, 
                     grid_cols: int = 3):
    
    num_images = len(images)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(grid_rows * grid_cols):
        ax = axes[i]
        if i < num_images:
            img = images[i].permute(1, 2, 0) 
            true_label = class_names[true_labels[i]]
            pred_label = class_names[predicted_labels[i]]

            ax.imshow(img)
            ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        else:
            pass

        ax.axis("off")

    plt.tight_layout()
    plt.show()