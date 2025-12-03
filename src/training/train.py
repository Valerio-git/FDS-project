from src.training.train_utils import train_model

if __name__ == "__main__":
    train_model(batch_size = 32, num_epochs = 10, learning_rate = 1e-3)