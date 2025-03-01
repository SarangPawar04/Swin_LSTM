import os
import torch
from lstm_model import TemporalLSTM

# Load trained LSTM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = TemporalLSTM().to(device)
lstm_model.load_state_dict(torch.load("models/lstm_model_custom.pth"))
lstm_model.eval()

def detect_deepfake(features_file):
    """Returns confidence score instead of just a label"""
    features = torch.load(features_file).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = lstm_model(features)
        confidence = torch.softmax(prediction, dim=1)[0, 1].item()  # Fake class probability
    return confidence  # Return probability instead of label

def detect_video(test_faces_folder, threshold=0.4, min_fake_frames=3):
    """Classifies video as deepfake based on an adaptive threshold"""
    deepfake_confidences = []
    total_faces = 0

    for face_feature in os.listdir(test_faces_folder):
        face_feature_path = os.path.join(test_faces_folder, face_feature)
        confidence = detect_deepfake(face_feature_path)
        deepfake_confidences.append(confidence)
        total_faces += 1

    # Compute average confidence & count of confident fake frames
    avg_confidence = sum(deepfake_confidences) / total_faces if total_faces > 0 else 0
    high_conf_fake_frames = sum(1 for c in deepfake_confidences if c > threshold)

    # If avg confidence is high or enough fake frames are detected, classify as deepfake
    if avg_confidence > threshold or high_conf_fake_frames >= min_fake_frames:
        return "Deepfake"
    return "Real"

# Example Usage
# print(detect_video("dataset/test_features/"))
