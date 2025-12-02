import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms 

from src.data.data_loader import WasteDataset
from src.models.CNN import CNN


# Dataset path
dataset_path = "/path/to/dataset"

# Transofrmations (same as traing part)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test-set
test_dataset = WasteDataset(dataset_path, split="test", transform=transform)

# Number of classes
num_classes = len(test_dataset.classes)

# Uploading the model
model = CNN(num_classes).to("cuda")

current_dir = os.path.dirname("src/testing")
model_path = os.path.join(current_dir, "..", "training", "best_model.pth")
model_path = os.path.abspath(model_path)

model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load("best_model.pth"))   # uploading best model found in training

# Perform sample inferences on random test images with different labels
model.eval()
with torch.no_grad():
    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    
    selected_images = []
    selected_labels = []
    selected_predicted = []
    
    for index in indices:
        image, label = test_dataset[index]
        image = image.unsqueeze(0).to('cuda')
        
        output = model(image)
        _, predicted = torch.max(output, 1)
        
        if label not in selected_labels:
            selected_images.append(image)
            selected_labels.append(label)
            selected_predicted.append(predicted.item())
        
        if len(selected_labels) == 9:
            break
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(9):
        axes[i].imshow(selected_images[i].squeeze().cpu().permute(1, 2, 0))
        axes[i].set_title(f"True: {test_dataset.classes[selected_labels[i]]}\nPredicted: {test_dataset.classes[selected_predicted[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()