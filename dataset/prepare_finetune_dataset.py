"""
prepare_finetune_dataset.py — Prepare FakeAVCeleb for Fine-tuning
=================================================================
Organizes FakeAVCeleb dataset into real/fake folders for binary classification.

FakeAVCeleb Categories:
- RealVideo-RealAudio (RARA) → REAL
- FakeVideo-FakeAudio (FAFA) → FAKE
- FakeVideo-RealAudio (FARA) → FAKE
- RealVideo-FakeAudio (RAFA) → FAKE
"""

import os
import sys
import shutil
import random
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


def find_zip_files(search_dir: Path) -> list:
    """Find all zip files in directory."""
    return list(search_dir.glob("*.zip"))


def unzip_dataset(zip_path: Path, extract_to: Path) -> Path:
    """Unzip dataset and return extracted folder path."""
    print(f"📦 Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the root folder name inside zip
        root_folders = set()
        for name in zip_ref.namelist():
            parts = name.split('/')
            if parts[0]:
                root_folders.add(parts[0])
        
        zip_ref.extractall(extract_to)
    
    print(f"✅ Extracted to {extract_to}")
    
    # Return the extracted root folder
    if len(root_folders) == 1:
        return extract_to / list(root_folders)[0]
    return extract_to


def find_fakeavceleb_folders(root: Path) -> dict:
    """
    Find the 4 FakeAVCeleb category folders.
    Returns dict mapping category to folder path.
    """
    # Possible folder naming conventions
    category_patterns = {
        'real': ['RealVideo-RealAudio', 'RARA', 'real_real', 'RealVideo_RealAudio'],
        'fake_vv_aa': ['FakeVideo-FakeAudio', 'FAFA', 'fake_fake', 'FakeVideo_FakeAudio'],
        'fake_vv_ra': ['FakeVideo-RealAudio', 'FARA', 'fake_real', 'FakeVideo_RealAudio'],
        'fake_rv_fa': ['RealVideo-FakeAudio', 'RAFA', 'real_fake', 'RealVideo_FakeAudio'],
    }
    
    found_folders = {}
    
    # Search recursively for these folders
    for category, patterns in category_patterns.items():
        for pattern in patterns:
            # Try exact match
            matches = list(root.rglob(pattern))
            if matches:
                found_folders[category] = matches[0]
                break
            
            # Try case-insensitive
            for folder in root.rglob("*"):
                if folder.is_dir() and folder.name.lower() == pattern.lower():
                    found_folders[category] = folder
                    break
    
    return found_folders


def collect_videos(folder: Path, extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')) -> list:
    """Collect all video files from a folder recursively."""
    videos = []
    for ext in extensions:
        videos.extend(folder.rglob(f"*{ext}"))
        videos.extend(folder.rglob(f"*{ext.upper()}"))
    return list(set(videos))  # Remove duplicates


def create_finetune_dataset(
    source_root: Path,
    output_dir: Path,
    num_real: int = 1000,
    num_fake: int = 1000,
    copy_files: bool = True
):
    """
    Create balanced real/fake dataset for fine-tuning.
    
    Args:
        source_root: Path to extracted FakeAVCeleb dataset
        output_dir: Path to output dataset_fine folder
        num_real: Number of real videos to include
        num_fake: Number of fake videos to include (total from all 3 fake categories)
        copy_files: If True, copy files. If False, create symlinks.
    """
    
    # Find category folders
    print("\n🔍 Searching for FakeAVCeleb folders...")
    folders = find_fakeavceleb_folders(source_root)
    
    if 'real' not in folders:
        print("❌ Could not find RealVideo-RealAudio folder!")
        print(f"   Searched in: {source_root}")
        print("\n   Available folders:")
        for f in source_root.rglob("*"):
            if f.is_dir() and len(list(f.iterdir())) > 0:
                print(f"   - {f.relative_to(source_root)}")
        return False
    
    print("\n📂 Found folders:")
    for cat, path in folders.items():
        print(f"   {cat}: {path}")
    
    # Collect videos from each category
    print("\n📹 Collecting videos...")
    
    real_videos = collect_videos(folders['real'])
    print(f"   Real (RARA): {len(real_videos)} videos")
    
    fake_videos = []
    fake_categories = ['fake_vv_aa', 'fake_vv_ra', 'fake_rv_fa']
    
    for cat in fake_categories:
        if cat in folders:
            cat_videos = collect_videos(folders[cat])
            fake_videos.extend(cat_videos)
            print(f"   {cat}: {len(cat_videos)} videos")
    
    print(f"\n   Total Real: {len(real_videos)}")
    print(f"   Total Fake: {len(fake_videos)}")
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("❌ No videos found in one or more categories!")
        return False
    
    # Shuffle and select
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    # Adjust numbers if not enough videos
    actual_real = min(num_real, len(real_videos))
    actual_fake = min(num_fake, len(fake_videos))
    
    # For balance, use the minimum
    balanced_count = min(actual_real, actual_fake)
    
    print(f"\n⚖️ Creating balanced dataset:")
    print(f"   Using {balanced_count} real and {balanced_count} fake videos")
    
    selected_real = real_videos[:balanced_count]
    selected_fake = fake_videos[:balanced_count]
    
    # Create output directories
    real_output = output_dir / "real"
    fake_output = output_dir / "fake"
    
    real_output.mkdir(parents=True, exist_ok=True)
    fake_output.mkdir(parents=True, exist_ok=True)
    
    # Copy/link files
    print(f"\n📁 {'Copying' if copy_files else 'Linking'} files to {output_dir}...")
    
    # Process real videos
    print("\n   Processing REAL videos...")
    for i, video in enumerate(tqdm(selected_real, desc="   Real")):
        # Create unique filename
        new_name = f"real_{i:04d}{video.suffix}"
        dest = real_output / new_name
        
        if copy_files:
            shutil.copy2(video, dest)
        else:
            if dest.exists():
                dest.unlink()
            os.symlink(video, dest)
    
    # Process fake videos
    print("\n   Processing FAKE videos...")
    for i, video in enumerate(tqdm(selected_fake, desc="   Fake")):
        new_name = f"fake_{i:04d}{video.suffix}"
        dest = fake_output / new_name
        
        if copy_files:
            shutil.copy2(video, dest)
        else:
            if dest.exists():
                dest.unlink()
            os.symlink(video, dest)
    
    # Summary
    print(f"\n✅ Dataset created successfully!")
    print(f"   Output: {output_dir}")
    print(f"   Real videos: {len(list(real_output.glob('*')))}")
    print(f"   Fake videos: {len(list(fake_output.glob('*')))}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare FakeAVCeleb dataset for fine-tuning")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to FakeAVCeleb dataset (folder or zip)")
    parser.add_argument("--output", type=str, default="dataset_fine",
                        help="Output folder for processed dataset")
    parser.add_argument("--num-videos", type=int, default=1000,
                        help="Number of videos per class (real and fake)")
    parser.add_argument("--symlink", action="store_true",
                        help="Create symlinks instead of copying files")
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    output_dir = project_root / args.output
    
    # Find source dataset
    if args.source:
        source_path = Path(args.source)
    else:
        # Look for zip files or extracted folders in dataset/
        print("🔍 Looking for FakeAVCeleb dataset...")
        
        # Check for zip files first
        zip_files = find_zip_files(script_dir)
        if zip_files:
            print(f"   Found zip: {zip_files[0].name}")
            source_path = unzip_dataset(zip_files[0], script_dir / "extracted")
        else:
            # Look for extracted folders
            possible_roots = [
                script_dir / "FakeAVCeleb",
                script_dir / "fakeavceleb", 
                script_dir / "extracted",
                script_dir,
            ]
            
            source_path = None
            for root in possible_roots:
                if root.exists():
                    folders = find_fakeavceleb_folders(root)
                    if 'real' in folders:
                        source_path = root
                        break
            
            if source_path is None:
                print("❌ Could not find FakeAVCeleb dataset!")
                print(f"   Searched in: {script_dir}")
                print("\n   Please either:")
                print("   1. Place the zip file in the dataset/ folder")
                print("   2. Extract the dataset to dataset/FakeAVCeleb/")
                print("   3. Use --source to specify the path")
                sys.exit(1)
    
    print(f"\n📁 Source: {source_path}")
    print(f"📁 Output: {output_dir}")
    
    # Create the dataset
    success = create_finetune_dataset(
        source_root=source_path,
        output_dir=output_dir,
        num_real=args.num_videos,
        num_fake=args.num_videos,
        copy_files=not args.symlink
    )
    
    if success:
        print(f"\n🎉 Done! You can now run:")
        print(f"   python finetune_deepfake.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
