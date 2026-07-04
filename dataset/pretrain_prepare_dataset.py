import os
import shutil
from tqdm import tqdm

def organize_voxceleb(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    print("Scanning for videos and moving them...")
    
    # Count total files first for the progress bar
    file_list = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".mp4"):
                file_list.append(os.path.join(root, file))

    # Move and rename files
    for file_path in tqdm(file_list, desc="Processing videos"):
        # Split path to get IDs: .../mp4/id00017/_mjZ87sK6cA/00095.mp4
        path_parts = file_path.split(os.sep)
        
        # We grab the last three parts to create a unique name
        # Change these indices if your path depth is different
        video_name = path_parts[-1]      # 00095.mp4
        youtube_id = path_parts[-2]      # _mjZ87sK6cA
        speaker_id = path_parts[-3]      # id00017
        
        # Construct a unique filename to avoid overwriting
        new_filename = f"{speaker_id}_{youtube_id}_{video_name}"
        destination = os.path.join(target_dir, new_filename)
        
        # Move the file (shutil.copy if you want to keep the original)
        shutil.move(file_path, destination)

    print(f"\nSuccess! All videos are now in: {target_dir}")

if __name__ == "__main__":
    # Path where you unzipped the data
    SOURCE = "voxceleb_data/mp4" 
    
    # Where you want the final flat list of videos
    DESTINATION = "voxceleb/real"
    
    organize_voxceleb(SOURCE, DESTINATION)