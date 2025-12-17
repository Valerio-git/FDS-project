import matplotlib.pyplot as plt
import torch

train_color = "#0E4111"
val_color   = "#04CA0E"
epoch_color  = "#187D94"

# in train_hyperparameters.py must be added the saving of training history like this:
'''torch.save({"model_state_dict":model.state_dict(),
                        "batch_size": best_batch_size,
                        "learning_rate": best_lr,
                        "weight_decay": wd,
                        "training_history": history
                        }, "src/checkpoints/cnn_stage1_A.pth")'''

def plot_training_CNN_no_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/cnn_stage1_A.pth", map_location=device)
    history = checkpoint["training_history"]
    best_lr = checkpoint["learning_rate"]
    best_batch_size = checkpoint["batch_size"]
    best_weight_decay = checkpoint["weight_decay"]

    train_loss = history["train_loss"]
    train_acc  = history["train_acc"]
    val_loss   = history["val_loss"]
    val_acc    = history["val_acc"]
    val_f1     = history["val_f1"]

    epochs = range(1, len(train_loss) + 1)
    last_epoch = val_f1.index(max(val_f1)) + 1

    # loss
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_loss, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_loss, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=last_epoch, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CNN (raw dataset) loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_raw_dataset_loss.png", bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_acc, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_acc, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=last_epoch, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CNN (raw dataset) accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_raw_dataset_accuracy.png", bbox_inches="tight")
    plt.close()

    # F1 validation
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_f1, label="val F1", linewidth=1.6, color = val_color)
    plt.axvline(x=last_epoch, linestyle='--', label=r"$\mathrm{1^{\text{st}} / 2^{\text{nd}}\ step}$", linewidth=1.2, color = epoch_color)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1 macro (val)")
    plt.title("CNN (raw dataset) F1 macro validation")
    plt.grid(True)
    plt.savefig("cnn_raw_dataset_f1.png", bbox_inches="tight")
    plt.close()


def plot_training_CNN_white():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/white_cnn_stage1_A.pth", map_location=device)
    history = checkpoint["training_history"]
    best_lr = checkpoint["learning_rate"]
    best_batch_size = checkpoint["batch_size"]
    best_weight_decay = checkpoint["weight_decay"]

    train_loss = history["train_loss"]
    train_acc  = history["train_acc"]
    val_loss   = history["val_loss"]
    val_acc    = history["val_acc"]
    val_f1     = history["val_f1"]

    epochs = range(1, len(train_loss) + 1)
    last_epoch = val_f1.index(max(val_f1)) + 1

    # loss
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_loss, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_loss, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=last_epoch, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CNN (white dataset) loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_white_dataset_loss.png", bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_acc, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_acc, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=last_epoch, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CNN (white dataset) accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_white_dataset_accuracy.png", bbox_inches="tight")
    plt.close()

    # F1 validation
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_f1, label="val F1", linewidth=1.6, color = val_color)
    plt.axvline(x=last_epoch, linestyle='--', label=r"$\mathrm{1^{\text{st}} / 2^{\text{nd}}\ step}$", linewidth=1.2, color = epoch_color)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1 macro (val)")
    plt.title("CNN (white dataset) F1 macro validation")
    plt.grid(True)
    plt.savefig("cnn_white_dataset_f1.png", bbox_inches="tight")
    plt.close()

# in fine_tune_CNN.py must be added the saving of training history like this:
'''torch.save({"model_state_dict":model.state_dict(),
                        "training_history": history
                        }, "src/checkpoints/cnn_stage2.pth")'''

def plot_training_CNN_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/cnn_stage2.pth", map_location=device)
    history = checkpoint["training_history"]

    train_loss = history["train_loss"]
    train_acc  = history["train_acc"]
    val_loss   = history["val_loss"]
    val_acc    = history["val_acc"]
    val_f1     = history["val_f1"]

    epochs = range(1, len(train_loss) + 1)
    last_epoch = val_f1.index(max(val_f1)) + 1

    # loss
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_loss, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_loss, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=10, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CNN (finetuned) loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_finetuned_loss.png", bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_acc, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_acc, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=10, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CNN (finetuned) accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_finetuned_accuracy.png", bbox_inches="tight")
    plt.close()

    # F1 validation
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_f1, label="val F1", linewidth=1.6, color = val_color)
    plt.axvline(x=10, linestyle='--', label=r"$\mathrm{1^{\text{st}} / 2^{\text{nd}}\ step}$", linewidth=1.2, color = epoch_color)
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1 macro (val)")
    plt.title("CNN (finetuned) F1 macro validation")
    plt.grid(True)
    plt.savefig("cnn_finetuned_f1.png", bbox_inches="tight")
    plt.close()

# in fine_tune_CNN.py must be added the saving of training history like this:
'''torch.save({"model_state_dict":model.state_dict(),
                        "training_history": history
                        }, "src/checkpoints/resnet_stage2.pth")'''


def plot_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("src/checkpoints/resnet_stage2.pth", map_location=device)
    history = checkpoint["training_history"]

    train_loss = history["train_loss"]
    train_acc  = history["train_acc"]
    val_loss   = history["val_loss"]
    val_acc    = history["val_acc"]
    val_f1     = history["val_f1"]

    epochs = range(1, len(train_loss) + 1)
    last_epoch = val_f1.index(max(val_f1)) + 1

    # loss
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_loss, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_loss, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=10, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("ResNet50 loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("resnet50_loss.png", bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_acc, label="val", linewidth=1.6, color = val_color)
    plt.plot(epochs, train_acc, label="train", linewidth=1.6, color = train_color)
    plt.axvline(x=10, linestyle='--', label='best epoch', linewidth=1.2, color = epoch_color)
    #plt.axvline(x=last_epoch, color='r ', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("ResNet50 accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("resnet50_accuracy.png", bbox_inches="tight")
    plt.close()

    # F1 validation
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, val_f1, label="val F1", linewidth=1.6, color = val_color)
    plt.axvline(x=10, linestyle='--', label=r"$\mathrm{1^{\text{st}} / 2^{\text{nd}}\ step}$", linewidth=1.2, color = epoch_color)
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1 macro (val)")
    plt.title("ResNet50 F1 macro validation")
    plt.grid(True)
    plt.savefig("resnet50_f1.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_training_CNN_no_finetune()
    plot_training_CNN_white()
    plot_training_CNN_finetune()
    plot_resnet()