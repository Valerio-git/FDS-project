import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

def grad_cam_cnn(model, x, class_idx=None, target_module=None):
    model.eval()
    if target_module is None:
        target_module = model.bn3

    activations = {}
    gradients = {}

    def fwd_hook(m, inp, out):
        activations["A"] = out

    def bwd_hook(m, grad_in, grad_out):
        gradients["dA"] = grad_out[0]

    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred = int(probs.argmax(dim=1).item())

    if class_idx is None:
        class_idx = pred

    model.zero_grad(set_to_none=True)
    logits[0, class_idx].backward()

    A = activations["A"]          # [1,C,h,w]
    dA = gradients["dA"]          # [1,C,h,w]

    weights = dA.mean(dim=(2, 3), keepdim=True)                  # [1,C,1,1]
    cam = F.relu((weights * A).sum(dim=1, keepdim=True))         # [1,1,h,w]
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove(); h2.remove()
    return cam, pred, probs.detach().cpu()[0]

def denorm_imagenet(t):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (t * std + mean).clamp(0, 1)

def gradcam_grid(model, dataset, device, n=6, rows=2, cols=3, seed=0):
    assert n == rows * cols
    random.seed(seed)
    idxs = random.sample(range(len(dataset)), n)

    plt.figure(figsize=(cols*4.2, rows*4.2))

    for k, idx in enumerate(idxs):
        x, y = dataset[idx]                 # x: [3,H,W], y: int
        x1 = x.unsqueeze(0).to(device)      # [1,3,H,W]

        cam, pred, probs = grad_cam_cnn(model, x1, class_idx=None, target_module=model.bn3)

        # Picture per plot
        img = denorm_imagenet(x).permute(1,2,0).cpu().numpy()

        true_name = dataset.classes[y] if hasattr(dataset, "classes") else str(y)
        pred_name = dataset.classes[pred] if hasattr(dataset, "classes") else str(pred)
        p = float(probs[pred])

        ax = plt.subplot(rows, cols, k+1)
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.axis("off")
        ax.set_title(f"T: {true_name}\nP: {pred_name} ({p:.2f})")

    plt.tight_layout()
    plt.show()

