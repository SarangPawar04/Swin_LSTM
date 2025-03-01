import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load extracted features from stored `.pt` files
real_features = torch.load("dataset/extracted_features/real.pt")
fake_features = torch.load("dataset/extracted_features/fake.pt")

# Assign labels
real_labels = torch.zeros(real_features.shape[0], dtype=torch.long)  # 0 for real
fake_labels = torch.ones(fake_features.shape[0], dtype=torch.long)   # 1 for fake

print(real_features.shape, fake_features.shape)  


# Combine real and fake datasets
features = torch.cat([real_features, fake_features], dim=0)
labels = torch.cat([real_labels, fake_labels], dim=0)

# Create dataset class
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create DataLoader
dataset = FeatureDataset(features, labels)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define LSTM Model
class CustomLSTM(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, output_dim=2):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return self.softmax(out)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/lstm_model_custom.pth")
