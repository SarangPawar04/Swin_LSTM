import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from datetime import datetime
from lstm_model import DeepfakeLSTM
from swin_feature_extraction import SwinFeatureExtractor

def load_model(model_path, device):
    """Load the LSTM model and move it to the specified device."""
    model = DeepfakeLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(image_path, feature_extractor, model, device):
    """Process a single image and return prediction."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor.extract_features(image_tensor)
            # Ensure features are on the same device as the model
            features = features.to(device)
            # Get prediction
            prediction = model(features)
            confidence = prediction.item()
        
        return confidence
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(image_path)}: {str(e)}")
        return None

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize feature extractor and model
    feature_extractor = SwinFeatureExtractor()
    model = load_model("models/lstm_model_best.pth", device)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Process all images in the input directory
    input_dir = "data/test_faces"
    results = []
    real_count = 0
    fake_count = 0
    
    print(f"\nProcessing images from: {input_dir}")
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            confidence = process_image(image_path, feature_extractor, model, device)
            
            if confidence is not None:
                is_real = confidence < 0.5
                if is_real:
                    real_count += 1
                else:
                    fake_count += 1
                
                results.append({
                    "image": filename,
                    "prediction": "Real" if is_real else "Fake",
                    "confidence": confidence
                })
    
    # Calculate percentages
    total = real_count + fake_count
    if total > 0:
        real_percentage = (real_count / total) * 100
        fake_percentage = (fake_count / total) * 100
    else:
        real_percentage = fake_percentage = 0
    
    # Create results dictionary
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_percentage": real_percentage,
        "fake_percentage": fake_percentage,
        "details": results
    }
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/detection_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total images processed: {total}")
    print(f"Real: {real_count} ({real_percentage:.1f}%)")
    print(f"Fake: {fake_count} ({fake_percentage:.1f}%)")

if __name__ == "__main__":
    main()
