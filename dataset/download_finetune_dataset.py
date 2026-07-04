import gdown

# The unique ID from your Google Drive link
file_id = '1GaKvQe9EgVOmIR1sHyBsvv496VjwnkFt'
url = f'https://drive.google.com/uc?id={file_id}'

# Define the output filename
output = 'downloaded_file.zip' # You can change this to the actual file name/extension

def download_large_file():
    try:
        print("Starting download... This may take a while for large files.")
        
        # gdown automatically handles the "large file" warning/pop-up
        gdown.download(url, output, quiet=False)
        
        print(f"\nSuccess! File saved as: {output}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_large_file()