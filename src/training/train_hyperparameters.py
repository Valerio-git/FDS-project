from training.train_utils import train_model

def hyperparameter_search():
    batch_sizes = [16, 32, 64]
    learning_rates = [1e-4, 1e-3, 1e-2]
    num_epochs = 5  # per la ricerca, puoi tenerlo basso per risparmiare tempo

    results = []

    for batch_size in batch_sizes:
        for lr in learning_rates:
            print(f"\n=== Training with batch_size={batch_size}, lr={lr} ===")
            _, history, _ = train_model(
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=lr,
                model_save_path=f"best_model_bs{batch_size}_lr{lr}.pth"
            )

            best_val_loss = min(history["val_loss"])
            best_val_acc = max(history["val_acc"])

            results.append({
                "batch_size": batch_size,
                "learning_rate": lr,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            })

    # Mostra i risultati ordinati per validation loss
    results_sorted = sorted(results, key=lambda x: x["best_val_loss"])
    print("\n=== Risultati ordinati per best validation loss ===")
    for r in results_sorted:
        print(
            f"bs={r['batch_size']}, lr={r['learning_rate']}, "
            f"val_loss={r['best_val_loss']:.4f}, val_acc={r['best_val_acc']:.4f}"
        )

    return results_sorted

if __name__ == "__main__":
    results = hyperparameter_search()
    print("Migliore combinazione:", results[0])