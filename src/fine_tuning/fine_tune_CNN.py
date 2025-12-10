import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.fine_tuning.fine_tuning import(
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


def fine_tuning_cnn():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = get_white_dataset_path()

    checkpoint = torch.load("src/checkpoints/cnn_stage1_A.pth", map_location=device)
    best_lr = checkpoint["learning_rate"]
    best_batch_size = checkpoint["batch_size"]
    best_weight_decay = checkpoint["weight_decay"]

    train_dataset, val_dataset, test_dataset, \
        train_loader, val_loader, test_loader = get_dataloaders(dataset_path, batch_size = best_batch_size, white = True)

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
        lr = best_lr,
        weight_decay = best_weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -float("inf")

    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        print(
            f"[Step1 Epoch {epoch+1}/5] "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc 
            torch.save(model.state_dict(), "src/checkpoints/cnn_stage2.pth")

    model.load_state_dict(torch.load("src/checkpoints/cnn_stage2.pth", map_location=device))

    #Training of the model whit freezing just the 1st and 2nd convolutional layers
    unfreeze_last_conv_block(model)
    optimizer = torch.optim.Adam(
        get_trainable_parameters(model),
        lr = best_lr * 0.1,
        weight_decay = best_weight_decay
    )

    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        print(
            f"[Step2 Epoch {epoch+1}/5] "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc 
            torch.save(model.state_dict(), "src/checkpoints/cnn_stage2.pth")

if __name__ == "__main__":
    fine_tuning_cnn()