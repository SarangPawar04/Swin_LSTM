import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class DeepfakeFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        """
        Dataset class for loading chunked features.
        
        Args:
            feature_dir (str): Directory containing the feature files (real.pt and fake.pt)
        """
        self.real_features = torch.load(os.path.join(feature_dir, "real.pt"))
        self.fake_features = torch.load(os.path.join(feature_dir, "fake.pt"))
        
        # Create labels (1 for real, 0 for fake)
        self.real_labels = torch.ones(len(self.real_features))
        self.fake_labels = torch.zeros(len(self.fake_features))
        
        # Combine features and labels
        self.features = torch.cat([self.real_features, self.fake_features], dim=0)
        self.labels = torch.cat([self.real_labels, self.fake_labels], dim=0)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DeepfakeLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, dropout=0.5):
        """
        LSTM model for deepfake detection using chunked features.
        
        Args:
            input_size (int): Size of each feature chunk (default: 128)
            hidden_size (int): Number of features in the hidden state (default: 256)
            num_layers (int): Number of recurrent layers (default: 2)
            dropout (float): Dropout rate (default: 0.5)
        """
        super(DeepfakeLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Tanh()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_chunks, chunk_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, num_chunks, hidden_size*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, num_chunks, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # Final classification
        output = self.fc(context)
        return torch.sigmoid(output).squeeze()

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
            
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device="cuda", patience=7):
    """
    Train the LSTM model with early stopping.
    
    Args:
        model (DeepfakeLSTM): The LSTM model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (str): Device to train on ("cuda" or "cpu")
        patience (int): Number of epochs to wait before early stopping
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).float()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, "models/lstm_model_best.pth")
            print("✓ Saved best model checkpoint")
        
        print("-" * 50)
        
        # If early stopping triggered, break the training loop
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return best_model_state

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    feature_dir = "dataset/extracted_features"
    dataset = DeepfakeFeatureDataset(feature_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = DeepfakeLSTM().to(device)
    
    # Train model with early stopping
    best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        patience=7  # Early stopping patience
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()