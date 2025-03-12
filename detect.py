import torch
from PIL import Image
import os
from lstm_model import DeepfakeLSTM
from swin_feature_extraction import transform, swin_model, split_features_into_chunks
import argparse

def detect_deepfake(image_path, lstm_model_path="models/lstm_model_best.pth"):
    """
    Detect if an image is real or fake using the trained LSTM model.
    
    Args:
        image_path (str): Path to the input image
        lstm_model_path (str): Path to the trained LSTM model weights
        
    Returns:
        tuple: (prediction (0=fake, 1=real), confidence score)
    """
    print(f"\nProcessing image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check if model exists
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM model not found: {lstm_model_path}")
    
    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and initialize models
    print("Loading LSTM model...")
    lstm_model = DeepfakeLSTM().to(device)
    try:
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        print("✓ LSTM model loaded successfully")
    except Exception as e:
        raise Exception(f"Error loading LSTM model: {str(e)}")
    
    lstm_model.eval()
    
    # Load and preprocess image
    print("Loading and preprocessing image...")
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        print("✓ Image preprocessed successfully")
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")
    
    # Extract features using Swin Transformer
    print("Extracting features...")
    with torch.no_grad():
        try:
            features = swin_model.forward_features(img_tensor)
            if features.dim() == 4:
                features = features.mean(dim=[1, 2])  # GAP
            print("✓ Features extracted successfully")
            
            # Split features into chunks
            feature_tensor = features.squeeze()
            print(f"Feature tensor shape: {feature_tensor.shape}")
            
            chunked_features = split_features_into_chunks(feature_tensor)
            print(f"Chunked features shape: {chunked_features.shape}")
            
            chunked_features = chunked_features.unsqueeze(0)  # Add batch dimension
            print(f"Final input shape: {chunked_features.shape}")
            
            # Get prediction from LSTM
            print("Running LSTM prediction...")
            output = lstm_model(chunked_features)
            print(f"Raw LSTM output: {output}")
            
            # Convert to probability and get binary prediction
            prob = float(output.cpu().numpy())  # Convert to float
            print(f"Probability: {prob}")
            
            # Ensure probability is between 0 and 1
            prob = max(0, min(1, prob))
            
            # Get binary prediction (0 for fake, 1 for real)
            prediction = int(prob >= 0.5)
            
            # Calculate confidence score (distance from decision boundary)
            confidence = prob if prediction == 1 else (1 - prob)
            
            print(f"Final prediction: {'Real' if prediction == 1 else 'Fake'} (confidence: {confidence:.4f})")
            return prediction, confidence
            
        except Exception as e:
            raise Exception(f"Error during feature extraction or prediction: {str(e)}")

def process_folder(input_dir, output_file="results.txt"):
    """
    Process all face images in a directory and save results to a file.
    
    Args:
        input_dir (str): Directory containing face images to process
        output_file (str): Path to save results
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
    print(f"\nProcessing faces from: {input_dir}")
    results = []
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Process each image
    image_extensions = ('.jpg', '.jpeg', '.png')
    for img_name in sorted(os.listdir(input_dir)):
        if img_name.lower().endswith(image_extensions):
            img_path = os.path.join(input_dir, img_name)
            try:
                prediction, confidence = detect_deepfake(img_path)
                status = "Real" if prediction == 1 else "Fake"
                results.append(f"{img_name}: {status} (confidence: {confidence:.4f})")
                print(f"✓ Processed {img_name} - {status} (confidence: {confidence:.4f})")
            except Exception as e:
                print(f"❌ Error processing {img_name}: {str(e)}")
                results.append(f"{img_name}: Error - {str(e)}")
    
    # Save results
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect deepfakes in face images')
    parser.add_argument('--input', required=True, help='Input directory containing face images')
    parser.add_argument('--output', default='results.txt', help='Output file to save results')
    
    args = parser.parse_args()
    
    try:
        process_folder(args.input, args.output)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
