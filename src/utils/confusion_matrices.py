import torch

def confusion_matrix_CNN_raw():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/cnn_stage1_A.pth", map_location=device)
    history = checkpoint["training_history"]
    val_f1 = history["val_f1"]
    last_epoch = val_f1.index(max(val_f1)) + 1

    conf_mat = history["val_conf_mat"][last_epoch - 1]

    return conf_mat

def confusion_matrix_CNN_white():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/white_cnn_stage1_A.pth", map_location=device)
    history = checkpoint["training_history"]
    val_f1 = history["val_f1"]
    last_epoch = val_f1.index(max(val_f1)) + 1

    conf_mat = history["val_conf_mat"][last_epoch - 1]

    return conf_mat

def confusion_matrix_CNN_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/cnn_stage2.pth", map_location=device)
    history = checkpoint["training_history"]
    val_f1 = history["val_f1"]
    last_epoch = val_f1.index(max(val_f1)) + 1

    conf_mat = history["val_conf_mat"][last_epoch - 1]

    return conf_mat

def confusion_matrix_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/resnet_stage2.pth", map_location=device)
    history = checkpoint["training_history"]
    val_f1 = history["val_f1"]
    last_epoch = val_f1.index(max(val_f1)) + 1

    conf_mat = history["val_conf_mat"][last_epoch - 1]

    return conf_mat

if __name__ == "__main__":
    print(confusion_matrix_CNN_raw())
    print(confusion_matrix_CNN_white())
    print(confusion_matrix_CNN_finetune())
    print(confusion_matrix_resnet())