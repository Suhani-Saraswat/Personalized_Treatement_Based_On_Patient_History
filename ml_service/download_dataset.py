"""
Script to download and extract the Kaggle / UCI Diabetes 130-US Hospitals (1999-2008) dataset.
Dataset source: Humberto Brandao (Kaggle: brandao/diabetes) / UCI ML Repository.
"""

import os
import sys
import zipfile
import urllib.request
import ssl

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ZIP_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")
CSV_PATH = os.path.join(DATA_DIR, "diabetic_data.csv")

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists at: {CSV_PATH}")
        size_mb = os.path.getsize(CSV_PATH) / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")
        return True

    print(f"Downloading Diabetes 130-US Hospitals dataset from {ZIP_URL}...")
    
    # Bypass certificate issues if system cert store is missing UCI CA
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    with urllib.request.urlopen(req, context=ctx) as response, open(ZIP_PATH, 'wb') as out_file:
        total_size = int(response.getheader('Content-Length', 0))
        downloaded = 0
        block_size = 65536
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
                sys.stdout.flush()
    print("\nDownload complete. Extracting ZIP...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    print(f"Extracted successfully into {DATA_DIR}!")
    
    # Check if there was an inner zip (UCI sometimes has an inner zip or direct csv)
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".zip") and f != "dataset.zip":
                inner_zip = os.path.join(root, f)
                print(f"Extracting nested zip: {inner_zip}")
                with zipfile.ZipFile(inner_zip, 'r') as inner_ref:
                    inner_ref.extractall(DATA_DIR)
            if f == "diabetic_data.csv":
                print(f"Found diabetic_data.csv ({os.path.getsize(os.path.join(root, f)) / (1024*1024):.2f} MB)")
                
    if os.path.exists(ZIP_PATH):
        try:
            os.remove(ZIP_PATH)
        except Exception:
            pass
            
    return os.path.exists(CSV_PATH)

if __name__ == "__main__":
    success = download_and_extract()
    if success:
        print("Kaggle Diabetes Dataset integration download succeeded!")
    else:
        print("Failed to find or extract diabetic_data.csv")
