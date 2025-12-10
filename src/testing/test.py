from src.testing.test_utils import (
    get_device,
    build_transform,
    load_test_dataset,
    load_trained_model,
    select_random_samples,
    denormalize_images,
    plot_predictions,
)
from src.utils.functions import ask_model_type_from_console

def testing(num_samples: int = 6, grid_rows: int = 2, grid_cols: int = 3, model_type = "cnn", white = False):
    
    device = get_device()
    print(f"Using device: {device}")

    model_type, white = ask_model_type_from_console()
    print(f"User selected model: {model_type}")

    transform = build_transform()
    test_dataset = load_test_dataset(transform, white = white)
    num_classes = len(test_dataset.classes)

    model = load_trained_model(num_classes=num_classes, device=device, white = white, model_type=model_type)

    images, true_labels, predicted_labels = select_random_samples(
        dataset=test_dataset,
        model=model,
        device=device,
        num_samples=num_samples,
        ensure_different_labels=True)

    images_denorm = denormalize_images(images)

    plot_predictions(
        images=images_denorm,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=test_dataset.classes,
        grid_rows=grid_rows,
        grid_cols=grid_cols)


if __name__ == "__main__":
    testing()