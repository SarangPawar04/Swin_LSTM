import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from lstm_model import DeepfakeLSTM, DeepfakeFeatureDataset

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Early stopping to stop training when validation loss doesn't improve for a given patience.
        
        Args:
            patience (int): Number of epochs to wait before stopping (default: 7)
            min_delta (float): Minimum change in monitored value to qualify as an improvement (default: 0)
            verbose (bool): Whether to print messages (default: True)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
class LstmTrain:
    def __init__(self, model, train_loader, val_loader, device="cuda", 
                 num_epochs=50, learning_rate=0.001, patience=5, min_delta=0.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5)

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train(self):
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_losses, train_preds, train_labels = [], [], []

            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device).float()

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                train_preds.extend((outputs > 0.5).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Validation
            self.model.eval()
            val_losses, val_preds, val_labels = [], [], []

            with torch.no_grad():
                for features, labels in self.val_loader:
                    features, labels = features.to(self.device), labels.to(self.device).float()
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)

                    val_losses.append(loss.item())
                    val_preds.extend((outputs > 0.5).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            train_acc = accuracy_score(train_labels, train_preds)
            val_acc = accuracy_score(val_labels, val_preds)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # LR scheduling
            self.scheduler.step(val_loss)

                        # Early stopping logic based on val loss, val acc, and overfitting detection
            improvement = False

            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                improvement = True

            if len(self.history['val_acc']) == 0 or val_acc > max(self.history['val_acc']) + 0.01:
                improvement = True

            if improvement:
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()
                os.makedirs("models", exist_ok=True)
                torch.save(self.best_model_state, "models/lstm_model_best.pth")
                print("✓ Saved best model checkpoint (loss/accuracy improved)")
            else:
                self.epochs_without_improvement += 1
                print(f"⚠️ No improvement. Patience: {self.epochs_without_improvement}/{self.patience}")

            # Overfitting check
            if train_acc >= 0.99 and val_acc < train_acc - 0.10:
                print(f"⚠️ Overfitting suspected. Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")
                self.epochs_without_improvement += 1

            # Stop if patience is exceeded
            if self.epochs_without_improvement >= self.patience:
                print("🛑 Early stopping triggered (no improvement or overfitting).")
                break


            print("-" * 50)

        plot_training_history(self.history)
        return self.best_model_state
         
def plot_training_history(history, save_dir='results'):
    """Plot and save training history."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    feature_dir = "dataset/extracted_features"
    dataset = DeepfakeFeatureDataset(feature_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = DeepfakeLSTM()

    trainer = LstmTrain(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=50,
    learning_rate=0.001,
    patience=7,
    min_delta=0.0
    )


    trainer.train()
    print("Training completed!")
    print("Training history plot saved as 'results/training_history.png'")


if __name__ == "__main__":
    main()