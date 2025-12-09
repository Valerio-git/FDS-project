import torchvision.transforms as T

# ========== TRASFORMAZIONI BASE (senza augmentation) ==========
no_aug_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========== TRASFORMAZIONI CON AUGMENTATION (SOLO TRAINING) ==========
train_aug_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),                  # flip orizzontale
    T.RandomRotation(degrees=15),                   # rotazioni leggere
    T.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)                     # piccoli shift
    ),
    T.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),                                              # leggere variazioni di colore
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ========== TRASFORMAZIONI PER VALIDATION / TEST ==========
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
    Restituisce la trasformazione corretta in base allo split e all'uso di augmentation.

    - split = 'train'  e use_augmentation=True  → train_aug_transform
    - split = 'train'  e use_augmentation=False → no_aug_transform
    - split = 'val'/'test'                      → val_test_transform
    """
    if split == "train":
        if use_augmentation:
            return train_aug_transform
        else:
            return no_aug_transform
    else:
        return val_test_transform