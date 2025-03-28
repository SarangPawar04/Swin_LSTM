import os
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories for testing."""
    directories = [
        'data/videos/test_vids',
        'data/test_frames',
        'data/test_faces',
        'dataset/test_features',
        'results'
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
    print("🚀 Starting Deepfake Detection Testing Pipeline...")
    
    # Step 1: Create necessary directories
    print("\n📁 Creating necessary directories...")
    create_directories()
    
    # Check if test videos exist
    test_dir = "data/videos/test_vids"
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        print(f"❌ Error: No test videos found in {test_dir}")
        print("Please place test videos in data/videos/test_vids/ directory")
        return
    
    # Step 2: Extract frames from test videos
    if not run_command(
        "python extract_frames.py --input data/videos/test_vids --output data/test_frames",
        "Extracting frames from test videos"
    ):
        return
    
    # Step 3: Extract faces from frames
    if not run_command(
        "python extract_faces.py --input data/test_frames --output data/test_faces",
        "Extracting faces from frames"
    ):
        return
    
    # Step 4: Extract features using Swin Transformer (with --test flag)
    if not run_command(
        "python swin_feature_extraction.py --input data/test_faces --output dataset/test_features --test",
        "Extracting features from faces"
    ):
        return
    
    # Step 5: Run model evaluation
    if not run_command(
        "python evaluation.py",
        "Evaluating model performance"
    ):
        return
    
    # Step 6: Clearing test_frames content
    if not run_command(
        "python clear_contents.py --target data/test_frames", 
        "Clearing test_frames Content"
    ):
        return
    
    # Step 7: Clearing test_faces content
    if not run_command(
        "python clear_contents.py --target data/test_faces", 
        "Clearing test_faces Content"
    ):
        return

    
    print("\n✨ Testing pipeline completed successfully!")
    print("\nResults are saved in the 'results' directory:")
    print("- Detection results: results/detection_results_*.json")
    print("- Evaluation results: results/evaluation_results_*.json")

if __name__ == "__main__":
    main() 