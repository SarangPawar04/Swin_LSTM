import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence

class DeepfakeFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        """
        Dataset class for loading video-wise chunked features.
        Each .pt file corresponds to one video (shape: [T, 8, 128])

        Args:
            feature_dir (str): Directory containing feature files like real_video1.pt, fake_video2.pt
        """
        self.feature_paths = []
        self.labels = []

        for filename in sorted(os.listdir(feature_dir)):
            if filename.endswith(".pt"):
                label = 0 if "real" in filename.lower() else 1
                self.feature_paths.append(os.path.join(feature_dir, filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        path = self.feature_paths[idx]
        features = torch.load(path)  # shape: [T, 8, 128]
        label = self.labels[idx]
        return features, label

class DeepfakeLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=2, dropout=0.5):
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

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device="cuda"):
    """
    Train the LSTM model.
    
    Args:
        model (DeepfakeLSTM): The LSTM model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (str): Device to train on ("cuda" or "cpu")
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, "models/lstm_model_best.pth")
            print("✓ Saved best model checkpoint")
        
        print("-" * 50)
    
    return best_model_state

# The value of T is variable so therefore to add padding we use the function "collate_fn"
# included Flattening: [8, 128] → [1024] so that it matches PyTorch LSTM's expected input format
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = [s.view(s.size(0), -1).float() for s in sequences]  # [T, 1024]
    padded_sequences = pad_sequence(sequences, batch_first=True)    # [B, max_T, 1024]
    labels = torch.tensor(labels).float()
    return padded_sequences, labels

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = DeepfakeLSTM().to(device)
    
    # Train model
    best_model_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
