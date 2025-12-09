import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.resnet import create_resnet50
from fine_tuning.fine_tuning import (
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = get_white_dataset_path()

train_dataset, val_dataset, test_dataset, \
    train_loader, val_loader, test_loader = get_dataloaders(dataset_path, batch_size = 64, white = True)

model = create_resnet50(len(train_dataset.classes))


#Training of just the fully connected layers
setup_feature_extraction(model)
optimizer = torch.optim.Adam(
    get_trainable_parameters(model),
    lr = 0.001
)

criterion = nn.CrossEntropyLoss()
best_val_acc = -float("inf")

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(
        f"[Step1 Epoch {epoch+1}/5] "
        f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
        f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
    )
    if val_acc > best_val_acc:
         best_val_acc = val_acc 
         torch.save(model.state_dict(), "src/checkpoints/resnet_stage2.pth")

model.load_state_dict(torch.load("src/checkpoints/resnet_stage2.pth", map_location=device))

#Training of the model whit freezing just the 1st and 2nd convolutional layers
setup_fine_tuning_last_block(model)
optimizer = torch.optim.Adam(
    get_trainable_parameters(model),
    lr = 0.0001
)

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(
        f"[Step2 Epoch {epoch+1}/5] "
        f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | "
        f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
    )
    if val_acc > best_val_acc:
         best_val_acc = val_acc 
         torch.save(model.state_dict(), "src/checkpoints/resnet_stage2.pth")