import torch
import random
import numpy as np

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def ask_model_type_from_console():
    """
    Prompts the user from the console which model to use.
    Returns 'cnn' or 'resnet'.
    """
    while True:
        choice = input("Pick the model you want to test ([c]nn / [r]esnet): ").strip().lower()
        if choice in ("c", "cnn"):
            white = input("Is it trained on the white dataset? ([y]es / [n]o): ").strip().lower() in ("y", "yes")
            return "cnn", white
        elif choice in ("r", "resnet"):
            white = True
            return "resnet", white
        else:
            print("Invalid choice. Please digits 'c' for CNN or 'r' for ResNet.")
