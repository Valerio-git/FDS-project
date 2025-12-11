import torch
from src.models.CNN import CNN
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn


def extract_kernel_CNN_no_finetune(num_cols=8, title=""):
    model = CNN(num_classes=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    state_dict = torch.load("src/checkpoints/CNN_stage1_A.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    kernels_conv1 = model.conv1.weight.data.clone().cpu()

    kernels = kernels_conv1.clone()

    out_ch, in_ch, h, w = kernels.shape
    num_rows = (out_ch + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))

    for idx in range(out_ch):
        r = idx // num_cols
        c = idx % num_cols

        ax = axes[r][c] if num_rows > 1 else axes[c]

        ker = kernels[idx]

        # Normalizzazione per visualizzare
        ker = (ker - ker.min()) / (ker.max() - ker.min() + 1e-8)

        # caso RGB (3 canali)
        img = ker.permute(1, 2, 0).numpy()  # [H,W,3]
        ax.imshow(img)

        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    

def extract_kernel_resnet(num_cols=8, title=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_features, 7)
    model = model.to(device)

    kernels_conv1 = model.conv1.weight.data.clone().cpu()
    kernels = kernels_conv1.clone()


    out_ch, in_ch, h, w = kernels.shape
    num_rows = (out_ch + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))

    for idx in range(out_ch):
        r = idx // num_cols
        c = idx % num_cols

        ax = axes[r][c] if num_rows > 1 else axes[c]

        ker = kernels[idx]

        # Normalizzazione per visualizzare
        ker = (ker - ker.min()) / (ker.max() - ker.min() + 1e-8)

        # caso RGB (3 canali)
        img = ker.permute(1, 2, 0).numpy()  # [H,W,3]
        ax.imshow(img)

        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #extract_kernel_CNN_no_finetune(num_cols=8, title="Kernels CNN without fine-tuning")
    extract_kernel_resnet(num_cols=8, title="Kernels ResNet50 pretrained on ImageNet")