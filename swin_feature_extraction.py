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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_and_save_features(face_folder, feature_output_folder):
    """
    Extract features for real and fake faces and store them in a single file per category.

    Parameters:
        face_root_folder (str): Path to extracted faces (should contain 'real/' and 'fake/')
        feature_output_folder (str): Path where extracted features will be saved
    """
    for category in sorted(os.listdir(face_folder)):
        category_folder = os.path.join(face_folder, category)
        all_features = []

        for face in sorted(os.listdir(category_folder)):
            face_path = os.path.join(category_folder, face)
            
            if not os.path.exists(face_path):
                print(f"⚠️ Skipping missing file: {face_path}")
                continue

            img = Image.open(face_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = swin_model.forward_features(img)
                if features.dim() == 4:
                    features = features.mean(dim=[1, 2])  # GAP

            feature_tensor = features.squeeze()
            print("Feature shape:", features.shape)
            all_features.append(feature_tensor)

        if not all_features:  # Prevent error when saving
            print(f"❌ No features found for {category}, skipping save.")
            continue

        torch.save(torch.stack(all_features), os.path.join(feature_output_folder, f"{category}.pt"))
        print(f"✅ Features saved for {category} in {feature_output_folder}")


# Example Usage
extract_and_save_features("data/extracted_faces/", "dataset/extracted_features/")
