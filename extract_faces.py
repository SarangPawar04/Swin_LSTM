from facenet_pytorch import MTCNN
from PIL import Image
import os
import argparse

# Initialize MTCNN with Improved Settings
mtcnn = MTCNN(keep_all=True, min_face_size=10, thresholds=[0.6, 0.7, 0.7])

def extract_faces_training(frame_folder, output_folder, category):
    """
    Extract faces for training data (with real/fake categories)
    """
    category_output = os.path.join(output_folder, category)
    os.makedirs(category_output, exist_ok=True)
    
    category_input = os.path.join(frame_folder, category)
    if not os.path.exists(category_input):
        print(f"❌ {category} folder not found in {frame_folder}")
        return

    processed_images = 0
    print(f"\n📁 Processing {category} frames...")
    
    for frame in sorted(os.listdir(category_input)):
        frame_path = os.path.join(category_input, frame)
        try:
            img = Image.open(frame_path).convert("RGB")
            faces, _ = mtcnn.detect(img)
            
            if faces is None:
                print(f"🔸 No faces detected in {frame}. Skipping.")
                continue

            print(f"🟢 Processing {frame} - Detected {len(faces)} face(s)")

            for i, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face)
                face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
                face_path = os.path.join(category_output, f"{frame[:-4]}_face_{i}_{category}.jpg")
                face_crop.save(face_path)
                processed_images += 1
        except Exception as e:
            print(f"❌ Error processing {frame}: {e}")

    print(f"✅ {category.capitalize()} faces extracted and saved in {category_output}")
    print(f"📊 Total {category.capitalize()} Faces Saved: {processed_images}")

def extract_faces_testing(frame_folder, output_folder):
    """
    Extract faces for testing data (store in video-wise folders)
    """
    os.makedirs(output_folder, exist_ok=True)
    processed_images = 0
    
    print("\n📁 Processing test frames...")
    # Iterate over extracted frames (which are video-wise stored)
    for video_name in sorted(os.listdir(frame_folder)):
        video_frame_path = os.path.join(frame_folder, video_name)
        if not os.path.isdir(video_frame_path):
            continue  # Skip non-folder items

        video_output_folder = os.path.join(output_folder, video_name)  # ✅ Create folder for each video
        os.makedirs(video_output_folder, exist_ok=True)

        print(f"\n🎬 Processing frames from video: {video_name}")
        for frame in sorted(os.listdir(video_frame_path)):  # Iterate over extracted frames for this video
            frame_path = os.path.join(video_frame_path, frame)
            try:
                img = Image.open(frame_path).convert("RGB")
                faces, _ = mtcnn.detect(img)
                
                if faces is None:
                    print(f"🔸 No faces detected in {frame}. Skipping.")
                    continue

                print(f"🟢 Processing {frame} - Detected {len(faces)} face(s)")

                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = map(int, face)
                    face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224))

                    # ✅ Save inside the correct video-wise folder
                    face_path = os.path.join(video_output_folder, f"{frame[:-4]}_face_{i}.jpg")
                    face_crop.save(face_path)
                    processed_images += 1
            except Exception as e:
                print(f"❌ Error processing {frame}: {e}")

    print(f"✅ Test faces extracted and saved in {output_folder}")
    print(f"📊 Total Test Faces Saved: {processed_images}")


def process_frames(input_folder, output_folder):
    """
    Automatically detect if this is training or testing data and process accordingly
    """
    # Check if this is training data (has real/fake folders)
    if os.path.exists(os.path.join(input_folder, 'real')) and os.path.exists(os.path.join(input_folder, 'fake')):
        print("📁 Training mode detected (real/fake folders found)")
        os.makedirs(output_folder, exist_ok=True)
        for category in ['real', 'fake']:
            extract_faces_training(input_folder, output_folder, category)
    else:
        print("📁 Testing mode detected (processing all frames)")
        extract_faces_testing(input_folder, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract faces from frames')
    parser.add_argument('--input', required=True, help='Input folder containing frames')
    parser.add_argument('--output', required=True, help='Output folder for extracted faces')
    
    args = parser.parse_args()
    
    print(f"Processing frames from: {args.input}")
    print(f"Saving faces to: {args.output}")
    
    process_frames(args.input, args.output)
    print("\n✅ All frames processed!")
