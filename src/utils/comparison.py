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


import random
import torch
import torch.nn as nn
from typing import List, Tuple

def select_random_plastic_samples(
    dataset,
    model: nn.Module,
    device: torch.device,
    plastic_label: int,
    num_samples: int = 4,
    seed: int = 42
):
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

            x = image.to(device).unsqueeze(0)
            output = model(x)
            pred = output.argmax(dim=1).item()

            # prendiamo SOLO gli errori CNN
            if pred == label:
                continue

            selected_images.append(image.cpu())
            selected_labels.append(label)
            selected_predicted.append(pred)

            if len(selected_images) == num_samples:
                break

    return selected_images, selected_labels, selected_predicted

def plot_model_on_images(
    model: nn.Module,
    images,
    true_labels,
    dataset,
    device,
    grid_rows=2,
    grid_cols=2,
    title_prefix=""
):
    model.eval()
    predicted_labels = []

    with torch.no_grad():
        for img in images:
            x = img.to(device).unsqueeze(0)
            pred = model(x).argmax(dim=1).item()
            predicted_labels.append(pred)

    images_denorm = denormalize_images(images)

    print(f"\n{title_prefix}")
    plot_predictions(
        images=images_denorm,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=dataset.classes,
        grid_rows=grid_rows,
        grid_cols=grid_cols
    )

if __name__ == "__main__":

    num_samples = 4
    transform = build_transform()
    test_dataset, test_loader = load_test_dataset(
        transform, white=True, seed=42, num_workers=0
    )

    num_classes = len(test_dataset.classes)
    device = get_device()

    # ===== CNN =====
    cnn = load_trained_model(
        num_classes=num_classes,
        device=device,
        white=True,
        model_type="cnn"
    )

    images, true_labels, cnn_preds = select_random_plastic_samples(
        dataset=test_dataset,
        model=cnn,
        device=device,
        plastic_label=4,   # controlla che sia davvero "plastic"
        num_samples=num_samples
    )

    # Plot CNN
    plot_model_on_images(
        model=cnn,
        images=images,
        true_labels=true_labels,
        dataset=test_dataset,
        device=device,
        grid_rows=2,
        grid_cols=2,
        title_prefix="CNN – errori sulla classe PLASTIC"
    )

    # ===== RESNET =====
    resnet = load_trained_model(
        num_classes=num_classes,
        device=device,
        white=True,
        model_type="resnet"
    )

    # Plot ResNet sugli STESSI campioni
    plot_model_on_images(
        model=resnet,
        images=images,
        true_labels=true_labels,
        dataset=test_dataset,
        device=device,
        grid_rows=2,
        grid_cols=2,
        title_prefix="ResNet – sugli stessi errori della CNN"
    )
