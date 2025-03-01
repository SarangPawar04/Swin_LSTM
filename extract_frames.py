import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=5):
    """
    Extracts frames from a video at a specified frame rate.
    
    Parameters:
        video_path (str): Path to the input video.
        output_folder (str): Folder where extracted frames will be saved.
        frame_rate (int): Extract 1 frame every `frame_rate` frames.
    """
    # Determine if the video is real or fake
    label = "real" if "real" in video_path.lower() else "fake"
    labeled_output_folder = os.path.join(output_folder, label)
    
    os.makedirs(labeled_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(labeled_output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        success, frame = cap.read()
        frame_count += 1

    cap.read()
    print(f"✅ Extracted frames saved in {labeled_output_folder}")

# Example Usage
extract_frames("data/videos/fake/fake_01.mp4", "data/extracted_frames/")

