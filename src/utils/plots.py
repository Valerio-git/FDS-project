import matplotlib.pyplot as plt
import torch

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
    plt.figure()
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CNN (no finetune) loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_no_finetune_loss.png", bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CNN (no finetune) accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_no_finetune_accuracy.png", bbox_inches="tight")
    plt.close()

    # F1 validation
    plt.figure()
    plt.plot(epochs, val_f1, label="val F1")
    plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1 macro (val)")
    plt.title("CNN (no finetune) F1 macro validation")
    plt.grid(True)
    plt.savefig("cnn_no_finetune_f1.png", bbox_inches="tight")
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
    plt.figure()
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CNN (finetuned) loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_finetuned_loss.png", bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CNN (finetuned) accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_finetuned_accuracy.png", bbox_inches="tight")
    plt.close()

    # F1 validation
    plt.figure()
    plt.plot(epochs, val_f1, label="val F1")
    plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
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
    plt.figure()
    plt.plot(epochs, val_loss, label="val")
    plt.plot(epochs, train_loss, label="train")
    plt.axvline(x=10, color='r', linestyle='--')
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("ResNet50 loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("resnet50_loss.png", bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs, val_acc, label="val")
    plt.plot(epochs, train_acc, label="train")
    plt.axvline(x=10, color='r', linestyle='--')
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("ResNet50 accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("resnet50_accuracy.png", bbox_inches="tight")
    plt.close()

    # F1 validation
    plt.figure()
    plt.plot(epochs, val_f1, label="val F1")
    plt.axvline(x=10, color='r', linestyle='--')
    #plt.axvline(x=last_epoch, color='r', linestyle='--', label='best epoch')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("F1 macro (val)")
    plt.title("ResNet50 F1 macro validation")
    plt.grid(True)
    plt.show()
    plt.savefig("resnet50_f1.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    #plot_training_CNN_no_finetune()
    #plot_training_CNN_finetune()
    plot_resnet()