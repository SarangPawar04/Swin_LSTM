import os
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories for testing."""
    directories = [
        'detect/video',
        'detect/detect_frames',
        'detect/detect_faces',
        'detect/detect_features',
        'detect/results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def run_command(command, description):
    """Run a command and handle any errors."""
    print(f"\n{description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {str(e)}")
        return False

def main():
    print("🚀 Starting Deepfake Detection For Single Video...")
    
    # Step 1: Create necessary directories
    print("\n📁 Creating necessary directories...")
    create_directories()
    
    # Check if detect videos exist
    detect_dir = "detect/"
    if not os.path.exists(detect_dir) or not os.listdir(detect_dir):
        print(f"❌ Error: No videos found in {detect_dir}")
        print("Please place detection video in detect/video directory")
        return
    
    # Step 2: Extract frames from test videos
    if not run_command(
        "python extract_frames.py --input detect/ --output detect/detect_frames",
        "Extracting frames from detection video"
    ):
        return
    
    # Step 3: Extract faces from frames
    if not run_command(
        "python extract_faces.py --input detect/detect_frames --output detect/detect_faces",
        "Extracting faces from frames"
    ):
        return
    
    # Step 4: Extract features using Swin Transformer (with --test flag)
    if not run_command(
        "python swin_feature_extraction.py --input detect/detect_faces --output detect/detect_features",
        "Extracting features from faces"
    ):
        return
    
    # Step 5: Run deepfake detection
    if not run_command(
        "python detect.py --mode detect --input detect/detect_features --output detect/results",
        "Running deepfake detection"
    ):
        return
    
    print("\n✨ Testing pipeline completed successfully!")
    print("\nResults are saved in the 'results' directory:")
    print("- Detection results: detect/results/detection_results_*.json")

if __name__ == "__main__":
    main() 