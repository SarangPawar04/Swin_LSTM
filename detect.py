import os
import json
import torch
import argparse
from glob import glob
from datetime import datetime
from lstm_model import DeepfakeLSTM

def load_model(model_path, device):
    model = DeepfakeLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def detect_from_features(input_dir, output_dir, model, device):
    os.makedirs(output_dir, exist_ok=True)

    feature_files = glob(os.path.join(input_dir, "**", "*.pt"), recursive=True)
    results = []
    real_count = 0
    fake_count = 0

    print(f"\n📂 Processing features from: {input_dir}")
    for feature_path in feature_files:
        try:
            features = torch.load(feature_path, map_location=device)
            # Flatten if shape is [T, 8, 128]
            if features.ndim == 3 and features.shape[1:] == (8, 128):
                features = features.reshape(features.shape[0], -1)

            # Add batch dimension if missing
            if features.ndim == 2:
                features = features.unsqueeze(0)
            features = features.to(device)

            with torch.no_grad():
                prediction = model(features)
                confidence = prediction.item()

            is_real = confidence < 0.5
            if is_real:
                real_count += 1
            else:
                fake_count += 1

            relative_path = os.path.relpath(feature_path, input_dir)
            results.append({
                "feature": relative_path,
                "prediction": "Real" if is_real else "Fake",
                "confidence": confidence
            })
        except Exception as e:
            print(f"❌ Failed on {feature_path}: {e}")

    total = real_count + fake_count
    real_pct = (real_count / total) * 100 if total else 0
    fake_pct = (fake_count / total) * 100 if total else 0

    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": results
    }

    output_file = os.path.join(output_dir, f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n✅ Detection completed. Results saved to: {output_file}")
    print(f"Real: {real_count} ({real_pct:.2f}%) | Fake: {fake_count} ({fake_pct:.2f}%)")

    return 1 if fake_count > real_count else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake detection on pre-extracted features.")
    parser.add_argument("--mode", choices=["test", "detect"], default="detect")
    parser.add_argument("--input", type=str, default=None, help="Input directory containing extracted features.")
    parser.add_argument("--output", type=str, default="results", help="Directory to save detection results.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/lstm_model_best.pth"
    model = load_model(model_path, device)

    input_dir = args.input if args.input else ("data/test_features" if args.mode == "test" else "detect/detect_features")
    detect_from_features(input_dir, args.output, model, device)
