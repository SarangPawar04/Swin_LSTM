import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import cv2
try:
    from decord import VideoReader
    from decord import cpu
    DECORD_AVAILABLE = True
except ImportError:
    import cv2
    DECORD_AVAILABLE = False

def extract_frames(video_path, output_folder, frame_rate=5, is_test=False, video_wise=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if is_test:
        output_folder = os.path.join(output_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)
        frame_prefix = "frame"
    elif video_wise:
        label = "real" if "real" in video_path.lower() else "fake"
        output_folder = os.path.join(output_folder, label, video_name)
        os.makedirs(output_folder, exist_ok=True)
        frame_prefix = "frame"
    else:
        label = "real" if "real" in video_path.lower() else "fake"
        output_folder = os.path.join(output_folder, label)
        os.makedirs(output_folder, exist_ok=True)
        frame_prefix = f"frame_{label}_{video_name}"

    saved_count = 0

    if DECORD_AVAILABLE:
        vr = VideoReader(video_path, ctx=cpu())
        for i in range(0, len(vr), frame_rate):
            frame = vr[i].asnumpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 🔧 Fix color inversion
            frame_filename = os.path.join(output_folder, f"{frame_prefix}_{i}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
    else:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        success, frame = cap.read()
        while success:
            if frame_count % frame_rate == 0:
                frame_filename = os.path.join(output_folder, f"{frame_prefix}_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
            success, frame = cap.read()
            frame_count += 1
        cap.release()

    print(f"✅ Extracted {saved_count} frames from {video_path}")
    print(f"   Saved in: {output_folder}")

def process_video_wrapper(args):
    video_path, output_folder, frame_rate, is_test, video_wise = args
    try:
        extract_frames(video_path, output_folder, frame_rate, is_test, video_wise)
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(video_path)}: {str(e)}")

def process_videos(input_folder, output_folder, frame_rate=5, video_wise=False):
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    test_videos = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]
    if test_videos:
        print(f"\n📁 Processing {len(test_videos)} test videos:")
        args_list = [(os.path.join(input_folder, v), output_folder, frame_rate, True, video_wise) for v in test_videos]
        with ThreadPoolExecutor() as executor:
            executor.map(process_video_wrapper, args_list)
        return

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
        args_list = [(os.path.join(category_path, v), output_folder, frame_rate, False, video_wise) for v in videos]
        with ThreadPoolExecutor() as executor:
            executor.map(process_video_wrapper, args_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--input', required=True, help='Input folder containing videos')
    parser.add_argument('--output', required=True, help='Output folder for extracted frames')
    parser.add_argument('--frame-rate', type=int, default=5, help='Extract 1 frame every N frames')
    parser.add_argument('--video-wise', action='store_true', help='Group training frames by video inside real/fake folders')

    args = parser.parse_args()

    print(f"Processing videos from: {args.input}")
    print(f"Saving frames to: {args.output}")
    print(f"Frame rate: {args.frame_rate}")

    process_videos(args.input, args.output, args.frame_rate, args.video_wise)
    print("\n✅ All videos processed!")
