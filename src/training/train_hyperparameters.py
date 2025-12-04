from training.train_utils import train_model

import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import transforms

def hyperparameter_search_lr_bs():
    batch_sizes = [16, 32, 64]
    learning_rates = [1e-4, 1e-3, 1e-2]
    num_epochs = 50  # A lot, but early stopping will help
    results = []
    best_global_loss = float("inf")
    best_global_config = None

    for batch_size in batch_sizes:
        for lr in learning_rates:
            print(f"\n=== Training with batch_size={batch_size}, lr={lr} ===")
            model, history, _ = train_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=lr,
                model_save_path=None,
                early_stopping=True,
                patience=5,
                weight_decay=0.0
            )

            best_val_loss = min(history["val_loss"])
            best_val_acc = max(history["val_acc"])

            results.append({
                "batch_size": batch_size,
                "learning_rate": lr,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            })

            if best_val_loss < best_global_loss:
                best_global_loss = best_val_loss
                best_global_config = {
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                }
                torch.save(model.state_dict(), "best_model_lr_bs.pth")
                print("👉 New global best model saved to best_model_lr_bs.pth")

    # Mostra i risultati ordinati per validation loss
    results_sorted = sorted(results, key=lambda x: x["best_val_loss"])
    print("\n=== Risultati ordinati per best validation loss ===")
    for r in results_sorted:
        print(
            f"bs={r['batch_size']}, lr={r['learning_rate']}, "
            f"val_loss={r['best_val_loss']:.4f}, val_acc={r['best_val_acc']:.4f}"
        )

    return results_sorted

def hyperparameter_search_weight_decay(best_batch_size, best_lr):

    weight_decays = [0.0, 1e-5, 1e-4, 5e-4]
    num_epochs = 50 

    results = []
    best_global_loss = float("inf")
    best_global_config = None

    for wd in weight_decays:
        print(f"\n=== Training with bs={best_batch_size}, lr={best_lr}, weight_decay={wd} ===")

        model, history, _ = train_model(
            batch_size=best_batch_size,
            num_epochs=num_epochs,
            learning_rate=best_lr,
            model_save_path=None,       # nessun salvataggio per run
            early_stopping=True,
            patience=7,
            weight_decay=wd,
        )

        best_val_loss = min(history["val_loss"])
        best_val_acc = max(history["val_acc"])

        results.append({
            "batch_size": best_batch_size,
            "learning_rate": best_lr,
            "weight_decay": wd,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
        })

        if best_val_loss < best_global_loss:
            best_global_loss = best_val_loss
            best_global_config = {
                "batch_size": best_batch_size,
                "learning_rate": best_lr,
                "weight_decay": wd,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            }
            torch.save(model.state_dict(), "best_model.pth")
            print("👉 New global best (LR+BS+WD) saved to best_model.pth")

    results_sorted = sorted(results, key=lambda x: x["best_val_loss"])

    print("\n=== RISULTATI WEIGHT_DECAY (ordinati per best validation loss) ===")
    for r in results_sorted:
        print(
            f"wd={r['weight_decay']}, "
            f"val_loss={r['best_val_loss']:.4f}, val_acc={r['best_val_acc']:.4f}"
        )

    print("\nMigliore combinazione completa:", best_global_config)
    return results_sorted, best_global_config

if __name__ == "__main__":
    # STEP 1: cerca LR + batch size
    results_lr_bs, best_lr_bs = hyperparameter_search_lr_bs()
    print("\nMigliore combinazione LR+BS:", best_lr_bs)

    best_bs = best_lr_bs["batch_size"]
    best_lr = best_lr_bs["learning_rate"]

    # STEP 2: a LR+BS fissati, cerca il miglior weight_decay
    results_wd, best_full = hyperparameter_search_weight_decay(best_bs, best_lr)
    print("\nConfigurazione finale consigliata:", best_full)