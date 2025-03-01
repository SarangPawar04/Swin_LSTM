import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lstm_model import TemporalLSTM

# Load trained LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = TemporalLSTM().to(device)
lstm_model.load_state_dict(torch.load("models/lstm_model_custom.pth"))
lstm_model.eval()

# Load test features
real_features = torch.load("dataset/test_features/real.pt").to(device)
fake_features = torch.load("dataset/test_features/fake.pt").to(device)

# Assign ground truth labels
real_labels = torch.zeros(real_features.shape[0], dtype=torch.long).to(device)  # 0 for real
fake_labels = torch.ones(fake_features.shape[0], dtype=torch.long).to(device)   # 1 for deepfake

# Combine features and labels
test_features = torch.cat([real_features, fake_features], dim=0)
test_labels = torch.cat([real_labels, fake_labels], dim=0)

# Perform inference
with torch.no_grad():
    predictions = lstm_model(test_features)
    predicted_labels = torch.argmax(predictions, dim=1)

# Compute metrics
accuracy = accuracy_score(test_labels.cpu(), predicted_labels.cpu())
precision = precision_score(test_labels.cpu(), predicted_labels.cpu())
recall = recall_score(test_labels.cpu(), predicted_labels.cpu())
f1 = f1_score(test_labels.cpu(), predicted_labels.cpu())

print(f"✅ Model Evaluation Results:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
