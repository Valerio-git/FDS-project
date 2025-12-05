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
from src.training.train_utils import (
    train_one_epoch,
    evaluate
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Upload the model with best parameter found during training on 1st dataset
model = load_cnn_from_checkpoint(
    checkpoint_path="src/checkpoints/cnn_stage1_A.pth",
    num_classes=num_classes,
    device=device,
)


#Training of just the fully connected layers
freeze_all_except_classifier(model)
optimizer = torch.optim.Adam(
    get_trainable_parameters(model),
    lr=1e-3,
)


train_one_epoch(model, dataloader, criterion, optimizer)


#Training of the model whit freezing just the 1st and 2nd convolutional layers
unfreeze_last_conv_block(model)
optimizer = torch.optim.Adam(
    get_trainable_parameters(model),
    lr=1e-4,
)
