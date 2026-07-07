import os
import sys
import urllib.request
import zipfile

# Ensure the root project directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.preprocessing import filter_midi_files, generate_labels, print_dataset_info

def download_and_extract(url, extract_to, name="dataset"):
    print(f"[Download] Downloading {name} from {url}...")
    zip_path = f"temp_{name.replace(' ', '_')}.zip"
    
    def report(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading... {percent}%")
            sys.stdout.flush()
        
    urllib.request.urlretrieve(url, zip_path, reporthook=report)
    print("\n[Download] Download complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    os.remove(zip_path)
    print("[Download] Extraction complete.")

if __name__ == "__main__":
    print_dataset_info()
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    labels_file = "data/labels/labels.json"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    
    choice = input("Download MAESTRO now? (y/n, default n for manual add): ").lower().strip() or "n"
    if choice == "y":
        maestro_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
        download_and_extract(maestro_url, raw_dir, "MAESTRO")
    
    print("\n[INFO] Thêm dataset đa dạng:")
    print("  - Copy thêm MIDI từ Lakh / GigaMIDI / VGMusic / MidiCaps vào data/raw/")
    print("  - Tạo subfolders nếu muốn: data/raw/maestro, data/raw/game, data/raw/lakh")
    
    print("\n[Preprocessing] Filtering all MIDI in data/raw/ ...")
    filter_midi_files(raw_dir, processed_dir, verbose=True)
    
    print("\n[Labels] Generating improved labels (better instrument/genre detection)...")
    generate_labels(processed_dir, labels_file)
    
    print("\n✅ Bộ dataset đã được cập nhật! Sử dụng data đa dạng hơn để train chất lượng cao.")
    print("Chạy lại bước này sau khi thêm file MIDI mới.")
