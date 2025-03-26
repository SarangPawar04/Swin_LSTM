import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from timm import create_model
import argparse
import shutil

# Detect device (MPS for Mac M2, CUDA for GPUs, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Initialize Swin Transformer model
swin_model = create_model('swin_base_patch4_window7_224', pretrained=True)
swin_model.head = torch.nn.Identity()  # Remove classification head

# Load fine-tuned model weights
model_path = "models/swin_model_best.pth"
if os.path.exists(model_path):
    swin_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    print(f"✅ Loaded fine-tuned model weights from {model_path}")
else:
    print(f"⚠️ Warning: Fine-tuned model weights not found! Using default pretrained model.")

swin_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Constants for feature chunking
NUM_CHUNKS = 8
CHUNK_SIZE = 128  # 1024 // 8 = 128

def split_features_into_chunks(features):
    """
    Split a 1024-dimensional feature vector into 8 chunks of 128 features each.
    
    Args:
        features (torch.Tensor): Input feature tensor of shape (1024,)
    
    Returns:
        torch.Tensor: Reshaped tensor of shape (8, 128)
    """
    return features.reshape(NUM_CHUNKS, CHUNK_SIZE)

def extract_features(input_dir, output_dir, is_test=False):
    """
    Extract features from face images using Swin Transformer.
    
    Args:
        input_dir (str): Directory containing face images
        output_dir (str): Directory to save extracted features
        is_test (bool): Whether processing test data
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    swin_model.to(device)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    features_list = []
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    # If test data process videowise faces
    if is_test:
        for video_name in sorted(os.listdir(input_dir)):
            video_path = os.path.join(input_dir, video_name)
            if not os.path.isdir(video_path):  
                continue  # Skip if not a directory (ignore stray files)
            features_list = []

            for img_name in sorted(os.listdir(video_path)):
                if img_name.lower().endswith(image_extensions):
                    img_path = os.path.join(video_path, img_name)
                    try:
                        # Load and preprocess image
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0).to(device)
                        
                        # Extract features
                        with torch.no_grad():
                            features = swin_model.forward_features(img_tensor)
                            if features.dim() == 4:
                                features = features.mean(dim=[1, 2])  # GAP
                            
                            # Split features into chunks
                            feature_tensor = features.squeeze()
                            chunked_features = split_features_into_chunks(feature_tensor)
                        
                        features_list.append(chunked_features)
                        print(f"✓ Processed {img_name}")
                        
                    except Exception as e:
                        print(f"❌ Error processing {img_name}: {str(e)}")
                        continue
            
            if features_list:
                # Stack all chunked features
                all_features = torch.stack(features_list)
                output_file = os.path.join(output_dir, f"{video_name}.pt")
                torch.save(all_features, output_file)
                print(f"\n✅ Features saved for test data in {output_file}")
    
    # If training data, process real and fake directories separately
    else:
        for class_name in ['real', 'fake']:
            class_dir = os.path.join(input_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ Directory not found: {class_dir}")
                continue
            
            # 🔁 New: loop over videos inside real/fake
            for video_name in sorted(os.listdir(class_dir)):
                video_path = os.path.join(class_dir, video_name)
                if not os.path.isdir(video_path):
                    continue
                
                features_list = []
                for img_name in sorted(os.listdir(video_path)):
                    if img_name.lower().endswith(image_extensions):
                        img_path = os.path.join(video_path, img_name)
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img_tensor = transform(img).unsqueeze(0).to(device)

                            with torch.no_grad():
                                features = swin_model.forward_features(img_tensor)
                                if features.dim() == 4:
                                    features = features.mean(dim=[1, 2])
                                feature_tensor = features.squeeze()
                                chunked_features = split_features_into_chunks(feature_tensor)

                            features_list.append(chunked_features)
                            print(f"✓ Processed {class_name}/{video_name}/{img_name}")

                        except Exception as e:
                            print(f"❌ Error processing {img_name}: {str(e)}")
                            continue
                
                if features_list:
                    all_features = torch.stack(features_list)
                    output_file = os.path.join(output_dir, f"{class_name}_{video_name}.pt")
                    torch.save(all_features, output_file)
                    print(f"✅ Saved features: {output_file}")

class SwinFeatureExtractor:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = swin_model.to(self.device)
        self.model.eval()
    
    def extract_features(self, image_tensor):
        """Extract features from an image tensor."""
        with torch.no_grad():
            features = self.model.forward_features(image_tensor)
            if features.dim() == 4:
                features = features.mean(dim=[1, 2])  # GAP
            
            # Split features into chunks
            feature_tensor = features.squeeze()
            chunked_features = split_features_into_chunks(feature_tensor)
            chunked_features = chunked_features.unsqueeze(0)  # Add batch dimension
            
            return chunked_features.to(self.device)
        
def clear_folder_contents(folder):
    """Deletes all files in the 'real' and 'fake' subdirectories of the given folder."""
    for subfolder in ['real', 'fake']:
        path = os.path.join(folder, subfolder)
        if os.path.exists(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features using Swin Transformer')
    parser.add_argument('--input', required=True, help='Input directory containing face images')
    parser.add_argument('--output', required=True, help='Output directory to save features')
    parser.add_argument('--test', action='store_true', help='Process test data')
    
    args = parser.parse_args()
    
    try:
        extract_features(args.input, args.output, args.test)

        # ✅ Clean up only for training data
        if not args.test:
            if os.path.exists(args.input):
                print(f"\n🧹 Cleaning up extracted faces content in: {args.input}")
                clear_folder_contents(args.input)

            # Infer frames path
            possible_frames_dir = os.path.join(os.path.dirname(args.input), "extracted_frames")
            if os.path.exists(possible_frames_dir):
                print(f"🧹 Cleaning up extracted frames content in: {possible_frames_dir}")
                clear_folder_contents(possible_frames_dir)

    except Exception as e:
        print(f"❌ Error: {str(e)}")