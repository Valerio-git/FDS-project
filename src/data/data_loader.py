import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.data_utils import get_raw_dataset_path
from src.data_utils import get_white_dataset_path


# Define the dataset class (modified to include a split parameter)
class WasteDataset(Dataset):
    def __init__(self, split, transform=None, white = False):
        if white:
            root_dir = get_white_dataset_path()
        else:
            root_dir = get_raw_dataset_path()
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        
        if not white:
            for i, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                for subfolder in ['default', 'real_world']:
                    subfolder_dir = os.path.join(class_dir, subfolder)
                    image_names = os.listdir(subfolder_dir)
                    random.shuffle(image_names)
                    
                    if split == 'train':
                        image_names = image_names[:int(0.6 * len(image_names))]
                    elif split == 'val':
                        image_names = image_names[int(0.6 * len(image_names)):int(0.8 * len(image_names))]
                    else:  # split == 'test'
                        image_names = image_names[int(0.8 * len(image_names)):]
                    
                    for image_name in image_names:
                        self.image_paths.append(os.path.join(subfolder_dir, image_name))
                        self.labels.append(i)
        else:
             for i, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                subfolder_dir = os.path.join(class_dir, 'real_world')
                image_names = os.listdir(subfolder_dir)
                random.shuffle(image_names) 
                if split == 'train':
                    image_names = image_names[:int(0.6 * len(image_names))]
                elif split == 'val':
                    image_names = image_names[int(0.6 * len(image_names)):int(0.8 * len(image_names))]
                else:  # split == 'test'
                    image_names = image_names[int(0.8 * len(image_names)):]
                
                for image_name in image_names:
                    self.image_paths.append(os.path.join(subfolder_dir, image_name))
                    self.labels.append(i)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label