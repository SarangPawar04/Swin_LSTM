from facenet_pytorch import MTCNN
from PIL import Image
import os

# ✅ Initialize MTCNN with Improved Settings
mtcnn = MTCNN(keep_all=True, min_face_size=10, thresholds=[0.6, 0.7, 0.7])

def extract_faces(frame_folder, output_folder):
    """
    Detects and extracts faces from frames using MTCNN.
    
    Parameters:
        frame_folder (str): Folder containing extracted frames.
        output_folder (str): Folder where cropped faces will be saved.
    """
    # ✅ Assign Labels Based on Folder Name
    label = "real" if "real" in frame_folder.lower() else "fake"
    labeled_output_folder = os.path.join(output_folder, label)
    os.makedirs(labeled_output_folder, exist_ok=True)

    processed_images = 0  # Counter for successful face detections

    for frame in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame)

        try:
            # ✅ Load Image with Exception Handling
            img = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error reading {frame_path}: {e}")
            continue  # Skip to the next image

        # ✅ Detect Faces
        faces, _ = mtcnn.detect(img)
        
        if faces is None:
            print(f"🔸 No faces detected in {frame}. Skipping.")
            continue  # No faces found, move to next frame

        print(f"🟢 Processing {frame} - Detected {len(faces)} face(s)")

        # ✅ Save Cropped Faces
        for i, face in enumerate(faces):
            try:
                x1, y1, x2, y2 = map(int, face)  # Convert to integer coordinates
                face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
                face_path = os.path.join(labeled_output_folder, f"{frame[:-4]}_face_{i}_{label}.jpg")

                face_crop.save(face_path)
                processed_images += 1
            except Exception as e:
                print(f"❌ Error processing face {i} in {frame}: {e}")

    print(f"✅ Faces extracted and saved in {labeled_output_folder}")
    print(f"📊 Total Faces Saved: {processed_images}")

# Example Usage
extract_faces("data/extracted_frames/real", "data/extracted_faces/")
