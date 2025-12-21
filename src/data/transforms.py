import torchvision.transforms as T

# ========== BASIC TRANSFORMATIONS (NO AUGMENTATION) ==========
no_aug_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========== TRANSFORMATIONS WITH AUGMENTATION (ONLY TRAINING) ==========
train_aug_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),                   # horizontal flip
    T.RandomRotation(degrees=15),                    # little rotation
    T.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)                       # small shifts
    ),
    T.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),                                               # slight color variations
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========== TRANSFORMATIONS FOR VALIDATION / TEST ==========

val_test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_transform(split: str = "train", use_augmentation: bool = True):
    """
    Returns the correct transformation based on the split and whether augmentation is used.

    - split = 'train' and use_augmentation=True → train_aug_transform
    - split = 'train' and use_augmentation=False → no_aug_transform
    - split = 'val'/'test' → val_test_transform
    """
    if split == "train":
        if use_augmentation:
            return train_aug_transform
        else:
            return no_aug_transform
    else:
        return val_test_transform