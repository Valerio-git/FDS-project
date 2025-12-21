import torch
import time

from src.utils.functions import (set_seed, get_device, ask_model_type_from_console)
from src.training.train_hyperparameters import (hyperparameter_search_lr_bs, 
                                                hyperparameter_search_weight_decay)
from src.fine_tuning.fine_tune_CNN import fine_tuning_cnn
from src.fine_tuning.fine_tune_resnet import fine_tuning_resnet
from src.testing.test import testing



def pipeline():
    
    print ("\n====  Beginning Pipeline  ====\n")
    num_workers = 0
    seed = 42
    time.sleep(1)
    print("[1] \n--- set seed ---\n")

    set_seed(seed)
    
    time.sleep(1)
    print("[2] \n--- get device ---\n")
    device = get_device()
    
    time.sleep(1)
    print("[3] \n--- define model--- \n")

    time.sleep(1)
    model_type, white = ask_model_type_from_console()

    time.sleep(1)
    print("[4] \n--- starting training phase ---\n")

    if not model_type == "resnet":
        results_lr_bs, best_lr_bs = hyperparameter_search_lr_bs(seed = seed, num_workers = num_workers)
        best_bs = best_lr_bs["batch_size"]
        best_lr = best_lr_bs["learning_rate"]
        results_wd, best_full = hyperparameter_search_weight_decay(best_bs, best_lr, seed = seed, num_workers = num_workers)
        time.sleep(1)
        print(f"\n best configuration obatined: {best_full}")
    
    if white == True:

        time.sleep(1)
        print("[5] \n--- fine tuning ---\n")
        if model_type == "cnn":
            fine_tuning_cnn(seed = seed, num_workers = num_workers)
        elif model_type == "resnet":
            fine_tuning_resnet(seed = seed, num_workers = num_workers)
    
    time.sleep(1)
    print("[6] \n--- starting testing phase ---\n")
    testing(num_samples = 6, grid_rows= 2, grid_cols = 3, white = white, model_type = model_type, seed = seed, num_workers = num_workers)
    if model_type != "resnet":
        testing(num_samples = 6, grid_rows= 2, grid_cols = 3, white = False, model_type = model_type, seed = seed, num_workers = num_workers)


if __name__ == "__main__":
    pipeline()
    
    
    

