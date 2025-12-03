from src.testing.test_utils import (
    get_device,
    build_transform,
    load_test_dataset,
    load_trained_model,
    select_random_samples,
    denormalize_images,
    plot_predictions,
)

def main(num_samples: int = 9, grid_rows: int = 3, grid_cols: int = 3):
    
    device = get_device()
    print(f"Using device: {device}")

    transform = build_transform()
    test_dataset = load_test_dataset(transform)
    num_classes = len(test_dataset.classes)

    model = load_trained_model(num_classes=num_classes, device=device)

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
    main()