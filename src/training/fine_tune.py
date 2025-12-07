import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.cnn_finetune import(
    load_cnn_from_checkpoint,
    freeze_all_except_classifier,
    unfreeze_last_conv_block,
    get_trainable_parameters
)
from src.data_utils import get_white_dataset_path
from src.training.train_utils import (
    get_dataloaders,
    train_one_epoch,
    evaluate
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = get_white_dataset_path()

train_dataset, val_dataset, test_dataset, \
    train_loader, val_loader, test_loader = get_dataloaders(dataset_path, batch_size = 32)

## number of batch size must be the optimal one

# Upload the model with best parameter found during training on 1st dataset
model = load_cnn_from_checkpoint(
    checkpoint_path = "src/checkpoints/cnn_stage1_A.pth",
    num_classes = len(train_dataset.classes),
    device=device,
)


#Training of just the fully connected layers
freeze_all_except_classifier(model)
optimizer = torch.optim.Adam(
    get_trainable_parameters(model),
    lr=1e-3,
)


criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(
        f"[Step1 Epoch {epoch+1}/5] "
        f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
        f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
    )


#Training of the model whit freezing just the 1st and 2nd convolutional layers
unfreeze_last_conv_block(model)
optimizer = torch.optim.Adam(
    get_trainable_parameters(model),
    lr=1e-4,
)

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(
        f"[Step2 Epoch {epoch+1}/5] "
        f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
        f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
    )