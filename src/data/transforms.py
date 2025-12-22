import torchvision.transforms as T

# BASIC TRANSFORMATIONS (NO AUGMENTATION)
no_aug_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# TRANSFORMATIONS WITH AUGMENTATION (ONLY TRAINING)
train_aug_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),                   
    T.RandomRotation(degrees=15),                    
    T.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)                       
    ),
    T.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),                                               
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# TRANSFORMATIONS FOR VALIDATION / TEST 

val_test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_transform(split: str = "train", use_augmentation: bool = True):
    
    if split == "train":
        if use_augmentation:
            return train_aug_transform
        else:
            return no_aug_transform
    else:
        return val_test_transform