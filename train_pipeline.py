import os
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/videos/real',
        'data/videos/fake',
        'data/extracted_frames/real',
        'data/extracted_frames/fake',
        'data/extracted_faces/real',
        'data/extracted_faces/fake',
        'dataset/extracted_features',
        'dataset/test_features',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def run_command(command, description):
    """Run a command and handle any errors."""
    print(f"\n{description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✓ {description} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {str(e)}")
        raise

def main():
    print("🚀 Starting Deepfake Detection Training Pipeline...")
    
    # Step 1: Create necessary directories
    print("\n📁 Creating necessary directories...")
    create_directories()
    
     ## SWIN TRAINING : 

    # Step 2: Extract frames from videos
    run_command("python extract_frames.py --input data/videos --output data/extracted_frames", 
                "Extracting frames from videos")
    
    # Step 3: Extract faces from frames
    run_command("python extract_faces.py --input data/extracted_frames --output data/extracted_faces", 
                "Extracting faces from frames")
    
    # Step 4: Train Swin Transformer model
    run_command("python train_swin.py", "Training Swin Transformer model")
    
    # Step 5: Extract features using Swin Transformer
    run_command("python clear_contents.py --target data/extracted_frames", 
                "Clearing Extracted_frames Content")
    
    # Step 5: Extract features using Swin Transformer
    run_command("python clear_contents.py --target data/extracted_faces", 
                "Clearing Extracted_faces Content")
    
    ## LSTM TRAINING :

    # Step 6: Extract frames from videos (organize video-wise)
    run_command("python extract_frames.py --input data/videos --output data/extracted_frames --video-wise", 
                "Extracting frames from videos (organizing video-wise)")
    
    # Step 7: Extract faces from frames (organize video-wise)
    run_command("python extract_faces.py --input data/extracted_frames --output data/extracted_faces --video-wise", 
                "Extracting faces from frames (organizing video-wise)")
    
    # Step 5: Extract features using Swin Transformer
    run_command("python swin_feature_extraction.py --input data/extracted_faces --output dataset/extracted_features/", 
                "Extracting features using Swin Transformer")
    
    # Step 5: Extract features using Swin Transformer
    run_command("python clear_contents.py --target data/extracted_frames", 
                "Clearing Extracted_frames Content")
    
    # Step 5: Extract features using Swin Transformer
    run_command("python clear_contents.py --target data/extracted_faces", 
                "Clearing Extracted_faces Content")

    # Step 6: Train LSTM model
    run_command("python train_lstm.py", "Training LSTM model")
    
    print("\n✨ Training pipeline completed successfully!")
    print("\nTrained models are saved in the 'models' directory:")
    print("- Swin Transformer model: models/swin_model_custom.pth")
    print("- LSTM model: models/lstm_model_custom.pth")

if __name__ == "__main__":
    main() 