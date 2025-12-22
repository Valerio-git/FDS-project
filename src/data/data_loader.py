import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.data_utils import get_raw_dataset_path
from src.data_utils import get_white_dataset_path
from PIL import Image

class WasteDataset(Dataset):
    def __init__(self, items, classes, transform=None):
        
        self.items = items
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(item["path"]).convert("RGB")
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
