from facenet_pytorch import MTCNN
from PIL import Image
import os
import argparse
import torch
from concurrent.futures import ThreadPoolExecutor

# Explicit device setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\U0001F527 Using device: {device}")

# Initialize MTCNN with Improved Settings and device
mtcnn = MTCNN(keep_all=True, min_face_size=10, thresholds=[0.6, 0.7, 0.7], device=device)

def process_single_frame(frame_path, output_template, category=None):
    try:
        img = Image.open(frame_path).convert("RGB")
        faces, _ = mtcnn.detect(img)

        if faces is None:
            print(f"\U0001F538 No faces detected in {os.path.basename(frame_path)}. Skipping.")
            return 0

        print(f"\U0001F7E2 Processing {os.path.basename(frame_path)} - Detected {len(faces)} face(s)")

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face)
            face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
            suffix = f"_{category}" if category else ""
            face_path = output_template.replace(".jpg", f"_face_{i}{suffix}.jpg")
            face_crop.save(face_path)
        return len(faces)
    except Exception as e:
        print(f"\u274C Error processing {os.path.basename(frame_path)}: {e}")
        return 0

def extract_faces_training(frame_folder, output_folder, category, video_wise=False):
    category_input = os.path.join(frame_folder, category)
    category_output = os.path.join(output_folder, category)
    os.makedirs(category_output, exist_ok=True)

    if not os.path.exists(category_input):
        print(f"\u274C {category} folder not found in {frame_folder}")
        return

    processed_images = 0
    print(f"\n\U0001F4C1 Processing {category} frames...")

    if video_wise:
        for video_name in sorted(os.listdir(category_input)):
            video_path = os.path.join(category_input, video_name)
            if not os.path.isdir(video_path):
                continue

            video_output_folder = os.path.join(category_output, video_name)
            os.makedirs(video_output_folder, exist_ok=True)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for frame in sorted(os.listdir(video_path)):
                    if not frame.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    frame_path = os.path.join(video_path, frame)
                    output_template = os.path.join(video_output_folder, frame)
                    futures.append(executor.submit(process_single_frame, frame_path, output_template, category))
                for f in futures:
                    processed_images += f.result()
    else:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for frame in sorted(os.listdir(category_input)):
                if not frame.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                frame_path = os.path.join(category_input, frame)
                output_template = os.path.join(category_output, frame)
                futures.append(executor.submit(process_single_frame, frame_path, output_template, category))
            for f in futures:
                processed_images += f.result()

    print(f"\u2705 {category.capitalize()} faces extracted and saved in {category_output}")
    print(f"\U0001F4CA Total {category.capitalize()} Faces Saved: {processed_images}")

def extract_faces_testing(frame_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    processed_images = 0

    print("\n\U0001F4C1 Processing test frames...")
    for video_name in sorted(os.listdir(frame_folder)):
        video_frame_path = os.path.join(frame_folder, video_name)
        if not os.path.isdir(video_frame_path):
            continue

        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        print(f"\n\U0001F3AC Processing frames from video: {video_name}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for frame in sorted(os.listdir(video_frame_path)):
                if not frame.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                frame_path = os.path.join(video_frame_path, frame)
                output_template = os.path.join(video_output_folder, frame)
                futures.append(executor.submit(process_single_frame, frame_path, output_template))
            for f in futures:
                processed_images += f.result()

    print(f"\u2705 Test faces extracted and saved in {output_folder}")
    print(f"\U0001F4CA Total Test Faces Saved: {processed_images}")

def process_frames(input_folder, output_folder, video_wise=False):
    if os.path.exists(os.path.join(input_folder, 'real')) and os.path.exists(os.path.join(input_folder, 'fake')):
        print("\U0001F4C1 Training mode detected (real/fake folders found)")
        os.makedirs(output_folder, exist_ok=True)
        for category in ['real', 'fake']:
            extract_faces_training(input_folder, output_folder, category, video_wise=video_wise)
    else:
        print("\U0001F4C1 Testing mode detected (processing all frames)")
        extract_faces_testing(input_folder, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract faces from frames')
    parser.add_argument('--input', required=True, help='Input folder containing frames')
    parser.add_argument('--output', required=True, help='Output folder for extracted faces')
    parser.add_argument('--video-wise', action='store_true', help='Group training faces by video inside real/fake folders')

    args = parser.parse_args()

    print(f"Processing frames from: {args.input}")
    print(f"Saving faces to: {args.output}")

    process_frames(args.input, args.output, args.video_wise)
    print("\n\u2705 All frames processed!")
