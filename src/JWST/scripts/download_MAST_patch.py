import csv
import os
import urllib.request

def download_sky_patches(csv_file_path, download_directory="."):
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    with open(csv_file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        
        # Filter for rows that meet our criteria first to get a total count
        valid_rows = [row for row in reader if len(row) >= 19 and row[18].strip() == "3"]
        
    total_files = len(valid_rows)
    for i, row in enumerate(valid_rows, 1):
        data_uri = row[6].strip()
        file_name = row[14].strip()
        
        download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"
        local_file_path = os.path.join(download_directory, file_name)
        
        if os.path.exists(local_file_path):
            print(f"[{i}/{total_files}] Skipping {file_name} (already downloaded)")
            continue
            
        print(f"[{i}/{total_files}] Downloading {file_name}...")
        try:
            urllib.request.urlretrieve(download_url, local_file_path)
            print(f"Successfully downloaded to {local_file_path}")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")

download_sky_patches("jwst_L3_sky_patches.csv", download_directory="/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/JWST_L3_sky_patches/")