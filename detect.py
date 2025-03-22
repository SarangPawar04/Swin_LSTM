import torch
from PIL import Image
import os
from lstm_model import DeepfakeLSTM
from swin_feature_extraction import transform, swin_model, split_features_into_chunks
import argparse
import json
from datetime import datetime

def detect_deepfake(image_path, lstm_model_path="models/lstm_model_best.pth"):
    """
    Detect if an image is real or fake using the trained LSTM model.
    
    Args:
        image_path (str): Path to the input image
        lstm_model_path (str): Path to the trained LSTM model weights
        
    Returns:
        tuple: (prediction (0=fake, 1=real), confidence score)
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check if model exists
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM model not found: {lstm_model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and initialize models
    lstm_model = DeepfakeLSTM().to(device)
    try:
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    except Exception as e:
        raise Exception(f"Error loading LSTM model: {str(e)}")
    
    lstm_model.eval()
    
    # Load and preprocess image
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")
    
    # Extract features using Swin Transformer
    with torch.no_grad():
        try:
            features = swin_model.forward_features(img_tensor)
            if features.dim() == 4:
                features = features.mean(dim=[1, 2])  # GAP
            
            # Split features into chunks
            feature_tensor = features.squeeze()
            chunked_features = split_features_into_chunks(feature_tensor)
            chunked_features = chunked_features.unsqueeze(0)  # Add batch dimension
            
            # Get prediction from LSTM
            output = lstm_model(chunked_features)
            prob = float(output.cpu().numpy())
            prob = max(0, min(1, prob))
            prediction = int(prob >= 0.5)
            confidence = prob if prediction == 1 else (1 - prob)
            
            return prediction, confidence
            
        except Exception as e:
            raise Exception(f"Error during feature extraction or prediction: {str(e)}")

def process_folder(input_dir="data/test_faces", output_dir="results"):
    """
    Process all face images in a directory and save results to JSON files.
    
    Args:
        input_dir (str): Directory containing face images to process (default: data/test_faces)
        output_dir (str): Directory to save results
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": 0,
        "real_count": 0,
        "fake_count": 0,
        "details": []
    }
    
    # Process each image
    image_extensions = ('.jpg', '.jpeg', '.png')
    for img_name in sorted(os.listdir(input_dir)):
        if img_name.lower().endswith(image_extensions):
            img_path = os.path.join(input_dir, img_name)
            try:
                prediction, confidence = detect_deepfake(img_path)
                status = "Real" if prediction == 1 else "Fake"
                
                # Update counts
                results["total_images"] += 1
                if prediction == 1:
                    results["real_count"] += 1
                else:
                    results["fake_count"] += 1
                
                # Add detailed result
                results["details"].append({
                    "image": img_name,
                    "prediction": status,
                    "confidence": float(confidence)
                })
                
                print(f"✓ Processed {img_name} - {status} (confidence: {confidence:.4f})")
                
            except Exception as e:
                print(f"❌ Error processing {img_name}: {str(e)}")
                results["details"].append({
                    "image": img_name,
                    "error": str(e)
                })
    
    # Calculate statistics
    if results["total_images"] > 0:
        results["real_percentage"] = (results["real_count"] / results["total_images"]) * 100
        results["fake_percentage"] = (results["fake_count"] / results["total_images"]) * 100
    
    # Save detailed results to JSON
    output_file = os.path.join(output_dir, f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n📊 Detection Summary:")
    print(f"Total images processed: {results['total_images']}")
    print(f"Real faces: {results['real_count']} ({results.get('real_percentage', 0):.1f}%)")
    print(f"Fake faces: {results['fake_count']} ({results.get('fake_percentage', 0):.1f}%)")
    print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect deepfakes in face images')
    parser.add_argument('--input', default='data/test_faces', help='Input directory containing face images')
    parser.add_argument('--output', default='results', help='Output directory to save results')
    
    args = parser.parse_args()
    
    try:
        process_folder(args.input, args.output)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
