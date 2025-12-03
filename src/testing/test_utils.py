import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms 

from src.data_utils import get_dataset_path
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


def load_test_dataset(transform: transforms.Compose) -> WasteDataset:
    dataset_path = get_dataset_path()
    return WasteDataset(dataset_path, split="test", transform=transform)


def get_best_model_path() -> str:
    current_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    model_path = os.path.join(base_dir, "best_model.pth")
    return model_path


def load_trained_model(num_classes: int, device: torch.device) -> CNN:
    model = CNN(num_classes).to(device)
    model_path = get_best_model_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_random_samples(dataset: WasteDataset, model: CNN, device: torch.device,
                           num_samples: int = 9,
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

