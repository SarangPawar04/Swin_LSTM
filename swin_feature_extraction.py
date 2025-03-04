import torch
from torchvision import transforms
from timm import create_model
from train_swin import CustomSwin
from PIL import Image
import os

# Load trained Swin model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model = create_model("swin_base_patch4_window7_224", num_classes=2, pretrained=False)
#swin_model = CustomSwin(num_classes=2)
swin_model.head = torch.nn.Identity()
swin_model.load_state_dict(torch.load("models/swin_model_best.pth"), strict=False)
swin_model.eval().to(device)

# Constants for feature chunking
NUM_CHUNKS = 8
CHUNK_SIZE = 128  # 1024 // 8 = 128

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def split_features_into_chunks(features):
    """
    Split a 1024-dimensional feature vector into 8 chunks of 128 features each.
    
    Args:
        features (torch.Tensor): Input feature tensor of shape (1024,)
    
    Returns:
        torch.Tensor: Reshaped tensor of shape (8, 128)
    """
    return features.reshape(NUM_CHUNKS, CHUNK_SIZE)

def extract_and_save_features(face_folder, feature_output_folder):
    """
    Extract features for real and fake faces and store them in a single file per category.
    Features are split into 8 chunks of 128 dimensions each to create a sequence-like structure.

    Parameters:
        face_root_folder (str): Path to extracted faces (should contain 'real/' and 'fake/')
        feature_output_folder (str): Path where extracted features will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(feature_output_folder, exist_ok=True)

    # Skip hidden files like .DS_Store
    categories = [f for f in sorted(os.listdir(face_folder)) if not f.startswith('.')]
    
    for category in categories:
        category_folder = os.path.join(face_folder, category)
        
        # Skip if not a directory
        if not os.path.isdir(category_folder):
            print(f"⚠️ Skipping non-directory: {category_folder}")
            continue
            
        all_features = []
        # Skip hidden files in the category folder
        faces = [f for f in sorted(os.listdir(category_folder)) if not f.startswith('.')]

        for face in faces:
            face_path = os.path.join(category_folder, face)
            
            if not os.path.isfile(face_path):
                print(f"⚠️ Skipping non-file: {face_path}")
                continue

            try:
                img = Image.open(face_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = swin_model.forward_features(img)
                    if features.dim() == 4:
                        features = features.mean(dim=[1, 2])  # GAP

                feature_tensor = features.squeeze()
                # Split features into chunks
                chunked_features = split_features_into_chunks(feature_tensor)
                print(f"✓ Processed {face} - Original shape: {feature_tensor.shape}, Chunked shape: {chunked_features.shape}")
                all_features.append(chunked_features)
                
            except Exception as e:
                print(f"❌ Error processing {face_path}: {str(e)}")
                continue

        if not all_features:  # Prevent error when saving
            print(f"❌ No features found for {category}, skipping save.")
            continue

        # Stack all chunked features
        # Final shape will be (num_samples, num_chunks, chunk_size)
        stacked_features = torch.stack(all_features)
        output_path = os.path.join(feature_output_folder, f"{category}.pt")
        torch.save(stacked_features, output_path)
        print(f"✅ Features saved for {category} in {output_path} with shape {stacked_features.shape}")


# Example Usage
extract_and_save_features("data/extracted_faces/", "dataset/extracted_features/")
