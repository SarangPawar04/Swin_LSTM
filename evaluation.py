import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lstm_model import DeepfakeLSTM
import json
from datetime import datetime
import os

def evaluate_model(model_path="models/lstm_model_best.pth", features_dir="dataset/test_features", output_dir="results"):
    """
    Evaluate the model's performance on test data.
    
    Args:
        model_path (str): Path to the trained LSTM model
        features_dir (str): Directory containing test features
        output_dir (str): Directory to save evaluation results
    """
    print("\n🔍 Starting Model Evaluation...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the model is trained and saved correctly.")
    
    # Load trained LSTM model
    print("Loading LSTM model...")
    lstm_model = DeepfakeLSTM().to(device)
    try:
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    lstm_model.eval()
    
    # Check if features directory exists
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    # Load test features
    print("Loading test features...")
    try:
        # Load test features from single file
        features_path = os.path.join(features_dir, "test_features.pt")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Test features file not found: {features_path}")
            
        test_features = torch.load(features_path).to(device)
        print("✓ Test features loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading test features: {str(e)}")
    
    # Perform inference
    print("\nRunning inference...")
    with torch.no_grad():
        predictions = lstm_model(test_features)
        predicted_labels = (predictions >= 0.5).long()
    
    # Calculate metrics
    # Assuming first half of test data is real (0) and second half is fake (1)
    num_samples = len(predicted_labels)
    true_labels = torch.cat([
        torch.zeros(num_samples // 2),
        torch.ones(num_samples - num_samples // 2)
    ]).to(device)
    
    # Convert to numpy for sklearn metrics
    predicted_labels_np = predicted_labels.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels_np, predicted_labels_np)
    precision = precision_score(true_labels_np, predicted_labels_np)
    recall = recall_score(true_labels_np, predicted_labels_np)
    f1 = f1_score(true_labels_np, predicted_labels_np)
    
    # Print metrics
    print("\n📊 Model Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Print predictions summary
    print(f"\n📊 Predictions Summary:")
    print(f"Total samples: {len(predicted_labels)}")
    print(f"Predicted real: {(predicted_labels == 0).sum().item()}")
    print(f"Predicted fake: {(predicted_labels == 1).sum().item()}")
    
    # Create results dictionary
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
            "total_samples": len(predicted_labels),
            "predicted_real": (predicted_labels == 0).sum().item(),
            "predicted_fake": (predicted_labels == 1).sum().item()
        }
    }
    
    # Save results to JSON
    output_file = os.path.join(output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    evaluate_model()
