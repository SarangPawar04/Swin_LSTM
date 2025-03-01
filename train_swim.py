import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from timm import create_model
import os

# ✅ Set Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix MPS Support for macOS (if applicable)
if torch.backends.mps.is_available():
    device = torch.device("mps")

# ✅ Define Swin Transformer Model (Using Pretrained Weights)
class CustomSwin(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomSwin, self).__init__()
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.swin(x)

# ✅ Data Transformations (Augmentation Added)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data Augmentation
    transforms.RandomRotation(10),  # Prevent Overfitting
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load Dataset
dataset = datasets.ImageFolder(root="data/extracted_faces", transform=transform)

# print(dataset.class_to_idx)  {real:0 _ fake:1}

# ✅ Split into Training (80%) and Validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ✅ Use Larger Batch Size (8 is too small)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ Initialize Model
model = CustomSwin(num_classes=2).to(device)

# ✅ Freeze Early Layers (Prevent Overfitting & Speed Up Training)
for name, param in model.named_parameters():
    if "layers.0" in name or "layers.1" in name:  # Freeze first 2 layers
        param.requires_grad = False

# ✅ Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-5)  # AdamW is better for transformers
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)  # Learning Rate Decay

# ✅ Training Loop with Validation & Best Model Saving
EPOCHS = 10
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # ✅ Validation Step
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

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    scheduler.step(val_loss)  # Adjust Learning Rate Based on Validation Loss

    # ✅ Save Best Model (Only When Validation Loss Improves)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/swin_model_best.pth")
        print("✅ Best Model Saved!")

print("✅ Training Complete!")
