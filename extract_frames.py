import cv2
import os
import argparse

def extract_frames(video_path, output_folder, frame_rate=5, is_test=False):
    """
    Extracts frames from a video at a specified frame rate.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Get video name without extension
    if is_test:
        #frames extracted in video specific folders.
        output_folder = os.path.join(output_folder, video_name) 
        os.makedirs(output_folder, exist_ok=True)
        frame_prefix = "frame"
    else:
        # For training videos, use real/fake folders
        label = "real" if "real" in video_path.lower() else "fake"
        output_folder = os.path.join(output_folder, label)
        os.makedirs(output_folder, exist_ok=True)
        frame_prefix = f"frame_{label}_{video_name}"

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_rate == 0:
            if is_test:
                frame_filename = os.path.join(output_folder, f"{frame_prefix}_{frame_count}.jpg")
            else:
                frame_filename = os.path.join(output_folder, f"{frame_prefix}_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print(f"✅ Extracted {saved_count} frames from {video_path}")
    print(f"   Saved in: {output_folder}")

def process_videos(input_folder, output_folder, frame_rate=5):
    """
    Process videos from real and fake folders.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Check if this is a test folder (contains videos directly)
    test_videos = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]
    if test_videos:
        print(f"\n📁 Processing {len(test_videos)} test videos:")
        for video in test_videos:
            video_path = os.path.join(input_folder, video)
            try:
                extract_frames(video_path, output_folder, frame_rate, is_test=True)
            except Exception as e:
                print(f"❌ Error processing {video}: {str(e)}")
        return

    # If no videos found directly, look for real/fake folders
    for category in ['real', 'fake']:
        category_path = os.path.join(input_folder, category)
        if not os.path.exists(category_path):
            print(f"❌ {category} folder not found in {input_folder}")
            continue
            
        videos = [f for f in os.listdir(category_path) if f.lower().endswith(video_extensions)]
        if not videos:
            print(f"❌ No videos found in {category_path}")
            continue
        
        print(f"\n📁 Processing {len(videos)} videos from {category} folder:")
        for video in videos:
            print(f"  - {video}")
            video_path = os.path.join(category_path, video)
            try:
                extract_frames(video_path, output_folder, frame_rate, is_test=False)
            except Exception as e:
                print(f"❌ Error processing {video}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--input', required=True, help='Input folder containing videos')
    parser.add_argument('--output', required=True, help='Output folder for extracted frames')
    parser.add_argument('--frame-rate', type=int, default=5, help='Extract 1 frame every N frames')
    
    args = parser.parse_args()
    
    print(f"Processing videos from: {args.input}")
    print(f"Saving frames to: {args.output}")
    print(f"Frame rate: {args.frame_rate}")
    
    process_videos(args.input, args.output, args.frame_rate)
    print("\n✅ All videos processed!")