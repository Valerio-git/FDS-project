from src.testing.test_utils import (
    get_device,
    build_transform,
    load_test_dataset,
    load_trained_model,
    select_random_samples,
    denormalize_images,
    plot_predictions,
)

def ask_model_type_from_console() -> str:
    """
    Prompts the user from the console which model to use.
    Returns 'cnn' or 'resnet'.
    """
    while True:
        choice = input("Pick the model you want to test ([c]nn / [r]esnet): ").strip().lower()
        if choice in ("c", "cnn"):
            white = input("Is it trained on the white dataset? ([y]es / [n]o): ").strip().lower() in ("y", "yes")
            return "cnn", white
        elif choice in ("r", "resnet"):
            white = True
            return "resnet", white
        else:
            print("Invalid choice. Please digits 'c' for CNN or 'r' for ResNet.")

def main(num_samples: int = 9, grid_rows: int = 3, grid_cols: int = 3):
    
    device = get_device()
    print(f"Using device: {device}")

    model_type, white = ask_model_type_from_console()
    print(f"User selected model: {model_type}")

    transform = build_transform()
    test_dataset = load_test_dataset(transform)
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
    main()