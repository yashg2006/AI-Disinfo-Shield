"""
Download microsoft/deberta-v3-base model files using requests (bypasses urllib SSL issue).
Files saved to: models/deberta-v3-base/
"""
import os
import requests
from tqdm import tqdm

SAVE_DIR = "models/deberta-v3-base"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://huggingface.co/microsoft/deberta-v3-base/resolve/main"

FILES = [
    ("config.json",            "config.json"),
    ("tokenizer_config.json",  "tokenizer_config.json"),
    ("spm.model",              "spm.model"),
    ("pytorch_model.bin",      "pytorch_model.bin"),
]

def download(filename, dest_name):
    url  = f"{BASE_URL}/{filename}"
    dest = os.path.join(SAVE_DIR, dest_name)
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size > 1000:
            print(f"  Already exists ({size/1e6:.1f} MB): {dest_name}")
            return
    print(f"\nDownloading: {filename}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest_name) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  Saved: {dest}")

for fname, dname in FILES:
    try:
        download(fname, dname)
    except Exception as e:
        print(f"  ERROR downloading {fname}: {e}")

print("\nAll downloads done. Files in:", SAVE_DIR)
import os
for f in os.listdir(SAVE_DIR):
    size = os.path.getsize(os.path.join(SAVE_DIR, f))
    print(f"  {f:35s}  {size/1e6:8.2f} MB")
