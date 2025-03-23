import os
import sys

def rename_videos(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return
    
    # Determine the prefix based on folder name
    if "real" in folder_path.lower():
        prefix = "real"
    elif "fake" in folder_path.lower():
        prefix = "fake"
    else:
        print("Folder name must contain 'real' or 'fake'.")
        return
    
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    videos = [f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
    videos.sort()  # Sort to maintain order
    
    for count, video in enumerate(videos):
        ext = os.path.splitext(video)[1]  # Get file extension
        new_name = f"{prefix}_{count:02d}{ext}"
        old_path = os.path.join(folder_path, video)
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {video} -> {new_name}")
    
    print("Renaming completed.")

if __name__ == "__main__":
    folder_path = "data/videos/real"
    rename_videos(folder_path)
