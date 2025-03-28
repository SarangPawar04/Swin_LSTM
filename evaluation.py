import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from lstm_model import DeepfakeLSTM
import seaborn as sns
import json
from datetime import datetime
import os

def evaluate_model(model_path="models/lstm_model_best.pth", features_dir="dataset/test_features", output_dir="results"):
    print("\n🔍 Starting Model Evaluation...")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print("Loading LSTM model...")
    lstm_model = DeepfakeLSTM().to(device)
    try:
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    lstm_model.eval()

    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    print("Loading test features...")
    video_names = []
    probs = []
    predicted_labels = []

    for filename in sorted(os.listdir(features_dir)):
        if filename.endswith(".pt"):
            video_path = os.path.join(features_dir, filename)
            try:
                features = torch.load(video_path).to(device)  # [T, 8, 128]
                features = features.reshape(features.shape[0], -1).unsqueeze(0)  # [1, T, 1024]
                with torch.no_grad():
                    output = lstm_model(features)
                    prob = output.item()
                    pred = int(prob >= 0.5)
                    probs.append(prob)
                    predicted_labels.append(pred)
                    video_names.append(filename)
                print(f"✓ Processed {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")

    if not predicted_labels:
        raise RuntimeError("No predictions generated!")

    num_samples = len(predicted_labels)
    # Assign true labels based on filename keywords (or modify as needed)
    true_labels = [0 if "real" in name.lower() else 1 for name in video_names]

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Print results
    print("\n📊 Model Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print(f"\n📊 Predictions Summary:")
    print(f"Total samples: {num_samples}")
    print(f"Predicted real: {predicted_labels.count(0)}")
    print(f"Predicted fake: {predicted_labels.count(1)}")

    # Save metrics + predictions
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "predictions": {
            "total_samples": num_samples,
            "predicted_real": predicted_labels.count(0),
            "predicted_fake": predicted_labels.count(1)
        },
        "processed_videos": video_names
    }

    output_file = os.path.join(output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Detailed results saved to: {output_file}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - LSTM')
    plt.legend(loc="lower right")

    roc_path = os.path.join(output_dir, f"roc_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"📈 ROC AUC curve saved to: {roc_path}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"📉 Confusion matrix saved to: {cm_path}")

    return results

if __name__ == "__main__":
    evaluate_model()
