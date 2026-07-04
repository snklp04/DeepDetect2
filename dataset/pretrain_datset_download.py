import requests
import zipfile
import os
from tqdm import tqdm

def download_and_unzip(url, extract_to='voxceleb_data'):
    # Define the local filename
    local_zip_filename = "vox2_test_mp4.zip"
    
    # 1. Download the file
    print(f"Starting download: {url}")
    try:
        response = requests.get(url, stream=True,verify=False)
        response.raise_for_status() # Check for request errors
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        with open(local_zip_filename, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        print(f"Download complete: {local_zip_filename}")

        # 2. Unzip the file
        print(f"Extracting to '{extract_to}'...")
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)

        with zipfile.ZipFile(local_zip_filename, 'r') as zip_ref:
            # Get list of files to show progress during extraction
            files = zip_ref.namelist()
            for file in tqdm(files, desc="Extracting"):
                zip_ref.extract(member=file, path=extract_to)
        
        print("Extraction complete.")

        # 3. Optional: Remove the zip file after extraction to save space
        # os.remove(local_zip_filename)
        # print(f"Removed temporary file {local_zip_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    DATASET_URL = "https://cn01.mmai.io/download/voxceleb?key=538ca0dc6042247c16eaa57488942e12378ea3dcf9469fbe71e0bfbeb034d749378397775cf30e0c8838bf699c36104fae1c30dec92cfbf2c4952f6cd7a57f37158c9e387dd644481cd3094c7bfa91d147db79c2cf2b947aa78754d848d29087313287d8215abcc5b9b61a7591e29b35fc145cfb96cfcc42a4a0b78d180652c7&file=vox2_test_mp4.zip"
    
    download_and_unzip(DATASET_URL)