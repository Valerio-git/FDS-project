import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet import create_resnet50
from src.fine_tuning.fine_tuning import (
    setup_feature_extraction,
    setup_fine_tuning_last_block,
    get_trainable_parameters
)
from src.data_utils import get_white_dataset_path
from src.training.train_utils import (
    get_dataloaders,
    train_one_epoch,
    evaluate
)


def fine_tuning_resnet(seed = 42, num_workers = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = get_white_dataset_path()

    train_dataset, val_dataset, \
        train_loader, val_loader = get_dataloaders(dataset_path, batch_size = 64, white = True, seed = seed, num_workers = num_workers, include_test = False)

    model = create_resnet50(len(train_dataset.classes)).to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_conf_mat":[]
    }
    #Training of just the fully connected layers
    setup_feature_extraction(model)
    optimizer = torch.optim.Adam(
        get_trainable_parameters(model),
        lr = 0.001
    )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = -float("inf")

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, val_f1, val_conf_mat = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_conf_mat"].append(val_conf_mat)

        print(
            f"[Step1 Epoch {epoch+1}/10] "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f} "
            f"Valf1={val_f1:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc 
            torch.save({"model_state_dict":model.state_dict(),
                        "training_history": history
                        }, "src/checkpoints/resnet_stage2.pth")
            
    state_dict = torch.load("src/checkpoints/resnet_stage2.pth", map_location = device)
    model.load_state_dict(state_dict["model_state_dict"])

    #Training of the model whit freezing just the 1st and 2nd convolutional layers
    setup_fine_tuning_last_block(model)
    optimizer = torch.optim.Adam(
        get_trainable_parameters(model),
        lr = 0.0001
    )

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _, val_f1, val_conf_mat = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)  
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_conf_mat"].append(val_conf_mat)

        print(
            f"[Step2 Epoch {epoch+1}/10] "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f} "
            f"Valf1={val_f1:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc 
            torch.save(model.state_dict(), "src/checkpoints/resnet_stage2.pth")

if __name__ == "__main__":
    fine_tuning_resnet()