import os
import shutil
import argparse

def clear_real_fake_structure(folder):
    """
    Clear contents of 'real' and 'fake' subfolders.
    """
    for subfolder in ['real', 'fake']:
        path = os.path.join(folder, subfolder)
        if os.path.exists(path):
            print(f"🧹 Clearing contents of: {path}")
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"❌ Error deleting {item_path}: {e}")
        else:
            print(f"⚠️ Skipping missing subfolder: {path}")

def clear_test_structure(folder):
    """
    Delete all video subfolders (used for test_faces or test_frames).
    """
    for subfolder in sorted(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            try:
                shutil.rmtree(subfolder_path)
                print(f"🗑️ Deleted: {subfolder_path}")
            except Exception as e:
                print(f"❌ Error deleting {subfolder_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear contents of 'real/fake' or test video folders under a target directory")
    parser.add_argument('--target', required=True, help='Top-level directory (e.g., extracted_faces, test_faces, extracted_frames, test_frames)')
    args = parser.parse_args()

    if not os.path.exists(args.target):
        print(f"❌ Target folder not found: {args.target}")
        exit(1)

    # Detect structure
    real_path = os.path.join(args.target, 'real')
    fake_path = os.path.join(args.target, 'fake')

    if os.path.isdir(real_path) or os.path.isdir(fake_path):
        print("📁 Detected training structure (real/fake)")
        clear_real_fake_structure(args.target)
    else:
        print("📁 Detected test structure (video-wise)")
        clear_test_structure(args.target)

    print("\n✅ Cleanup complete.")
