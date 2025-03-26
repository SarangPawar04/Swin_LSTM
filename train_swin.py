import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from timm import create_model
import os
import sys
# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
# ✅ Set Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix MPS Support for macOS (if applicable)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")

# ✅ Define Swin Transformer Model
class CustomSwin(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomSwin, self).__init__()
        self.swin = create_model('swin_base_patch4_window7_224', pretrained=True)
        self.num_features = self.swin.num_features
        # Remove the original head
        self.swin.head = nn.Identity()
        # Add global average pooling and new classification head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Get features from Swin
        x = self.swin.forward_features(x)
        # Reshape and pool
        B, _, _, C = x.shape
        x = x.reshape(B, C, -1)  # [B, C, H*W]
        x = self.avgpool(x)      # [B, C, 1]
        x = x.flatten(1)         # [B, C]
        # Classification
        x = self.classifier(x)   # [B, num_classes]
        return x

# Initialize model
swin_model = CustomSwin()

def split_features_into_chunks(features, chunk_size=8):
    return features.reshape(chunk_size, -1)

# ✅ Data Transformations (Augmentation Added)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# TRAINING CODE - Only run this if training
if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    #ImageFolder dataset, which automatically assigns labels based on the directory structure.
    dataset = datasets.ImageFolder(root="data/extracted_faces", transform=transform)
    #dataset variable contains all the images and their corresponding labels.
    print(dataset.class_to_idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # ✅ Use Larger Batch Size (8 is too small)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ✅ Initialize Model
    model = CustomSwin(num_classes=2).to(device)
    print("Model initialized")

    # ✅ Freeze Early Layers (Prevent Overfitting & Speed Up Training)
    for name, param in model.named_parameters():
        if "layers.0" in name or "layers.1" in name:  # Freeze first 2 layers
            param.requires_grad = False

    # ✅ Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)  # AdamW is better for transformers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)  # Learning Rate Decay

    # ✅ Training Loop with Validation & Best Model Saving
    EPOCHS = 50
    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0
    print("\nStarting training...")

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        train_accuracy = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)

        # Print epoch results
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping logic
        # Early stopping logic with val loss, val acc, and overfitting check
        if avg_val_loss < best_val_loss - 1e-4 or val_accuracy > best_val_acc + 0.1:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            epochs_without_improvement = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/swin_model_best.pth")
            print("✅ New best model saved!")
        else:
            # Check for overfitting: high train acc, low val acc
            if train_accuracy >= 99.0 and val_accuracy < train_accuracy - 10:
                print(f"⚠️ Overfitting suspected. Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
                epochs_without_improvement += 1
            else:
                print(f"⚠️ No improvement. Patience counter: {epochs_without_improvement+1}/{patience}")
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("\n🛑 Early stopping triggered due to no improvement or overfitting.")
            break

    print("\n✅ Training Complete!")

