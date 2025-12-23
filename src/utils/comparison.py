import torch
import torch.nn as nn
import os
import random

from src.testing.test_utils import (
    get_device,
    build_transform,
    load_test_dataset,
    evaluate_test,
    load_trained_model,
    select_random_samples,
    denormalize_images,
    plot_predictions,
)
from src.testing.test import testing
from src.data.data_loader import WasteDataset
from src.utils.functions import ask_model_type_from_console


def select_random_plastic_samples(
    dataset: WasteDataset,
    model: nn.Module,
    device: torch.device,
    plastic_label: int,
    num_samples: int = 2,
    seed: int = 122333
) -> tuple[list[torch.Tensor], list[int], list[int]]:

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    selected_images = []
    selected_labels = []
    selected_predicted = []

    model.eval()
    with torch.no_grad():
        for idx in indices:

            image, label = dataset[idx]

            if label != plastic_label:
                continue

            image = image.to(device).unsqueeze(0)

            output = model(image)
            _, predicted = torch.max(output, dim=1)
            predicted_label = predicted.item()

            selected_images.append(image.squeeze(0))
            selected_labels.append(label)
            selected_predicted.append(predicted_label)

            if len(selected_images) == num_samples:
                break

    return selected_images, selected_labels, selected_predicted

def testing_plastic(num_samples: int = 6, grid_rows: int = 2, grid_cols: int = 3, model_type = "cnn", white = False, seed = 42, num_workers = 0):
    
    device = get_device()
    print(f"Using device: {device}")

    print(f"User selected model: {model_type}")

    transform = build_transform()
    test_dataset, test_loader = load_test_dataset(transform, white = white, seed = seed, num_workers = num_workers)
    num_classes = len(test_dataset.classes)

    model = load_trained_model(num_classes=num_classes, device=device, white = white, model_type=model_type)

    test_acc, _, _, test_f1 = evaluate_test(model, test_loader, device=device)
    print(f"Test accuracy = {test_acc:.4f} | Test f1-score = {test_f1:.4f}")

    images, true_labels, predicted_labels = select_random_plastic_samples(
        dataset=test_dataset,
        model=model,
        device=device,
        plastic_label=4, 
        num_samples=num_samples)

    images_denorm = denormalize_images(images)

    plot_predictions(
        images=images_denorm,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=test_dataset.classes,
        grid_rows=grid_rows,
        grid_cols=grid_cols)

if __name__ == "__main__":

    testing_plastic(num_samples = 6, grid_rows = 3, grid_cols = 2, model_type = "cnn" , white = True)
    testing_plastic(num_samples = 6, grid_rows = 3, grid_cols = 2, model_type = "resnet" , white = True)