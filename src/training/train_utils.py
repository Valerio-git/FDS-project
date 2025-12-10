import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix

from src.data_utils import get_white_dataset_path
from src.data_utils import get_raw_dataset_path
from src.data.transforms import get_transform

from src.data.data_loader import WasteDataset
from src.models.CNN import CNN

#define which hardware to use (GPU o CPU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_dataloaders(dataset_path, batch_size, white = False):
    train_transform = get_transform(split="train", use_augmentation=True)
    val_test_transform = get_transform(split="val", use_augmentation=False)

    train_dataset = WasteDataset(dataset_path, split='train', transform=train_transform, white = white)
    val_dataset = WasteDataset(dataset_path, split='val', transform=val_test_transform, white = white)
    test_dataset = WasteDataset(dataset_path, split='test', transform=val_test_transform, white = white)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
           
            all_preds.extend(preds.cpu().numpy()) # takes preds tensor and pass them to cpu, extend converts them to lists
            all_labels.extend(labels.cpu().numpy()) # same process for labels

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_preds = np.array(all_preds) # coverting them into numpy arrays 
    all_labels = np.array(all_labels)

    return epoch_loss, epoch_acc, all_preds, all_labels

    
def train_model(white = False, batch_size = 32, num_epochs = 5, learning_rate = 1e-3, model_save_path=None, early_stopping = False, patience = 5, weight_decay = 0.0):
    
    if white:
        dataset_path = get_white_dataset_path()
    else:
        dataset_path = get_raw_dataset_path()

    train_dataset, val_dataset, test_dataset, \
        train_loader, val_loader, test_loader = get_dataloaders(dataset_path, batch_size, white = white)

    num_classes = len(train_dataset.classes)
    model = CNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_conf_mat":[]
    }

    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_conf_mat = confusion_matrix(val_labels, val_preds)

        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_conf_mat"].append(val_conf_mat)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val_f1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            if early_stopping:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("Training completed!")
    return model, history, (train_dataset, val_dataset, test_dataset)