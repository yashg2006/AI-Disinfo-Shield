"""
Resume download of pytorch_model.bin using HTTP Range headers.
"""
import os, requests
from tqdm import tqdm

SAVE_DIR = "models/deberta-v3-base"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://huggingface.co/microsoft/deberta-v3-base/resolve/main"

def resumable_download(filename, dest_name, max_retries=10):
    url  = f"{BASE_URL}/{filename}"
    dest = os.path.join(SAVE_DIR, dest_name)
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers  = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=120)
            if r.status_code == 416:
                print(f"  {dest_name}: already fully downloaded.")
                return
            r.raise_for_status()
            total = existing + int(r.headers.get("content-length", 0))
            print(f"\nDownloading {dest_name}  (attempt {attempt}, resuming from {existing/1e6:.1f} MB, total ~{total/1e6:.1f} MB)")
            mode = "ab" if existing > 0 else "wb"
            with open(dest, mode) as f, tqdm(total=total, initial=existing, unit="B", unit_scale=True, desc=dest_name) as bar:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"  Done: {dest}")
            return
        except Exception as e:
            existing = os.path.getsize(dest) if os.path.exists(dest) else 0
            headers  = {"Range": f"bytes={existing}-"}
            print(f"  Interrupted at {existing/1e6:.1f} MB (attempt {attempt}): {e}")
            print(f"  Retrying...")

    print(f"  FAILED after {max_retries} attempts.")

# Only resume pytorch_model.bin (config/tokenizer already done)
resumable_download("pytorch_model.bin", "pytorch_model.bin")

# Verify final size
dest = os.path.join(SAVE_DIR, "pytorch_model.bin")
final_size = os.path.getsize(dest) if os.path.exists(dest) else 0
print(f"\nFinal file size: {final_size/1e6:.2f} MB")
if final_size > 370_000_000:
    print("Download complete and looks correct!")
else:
    print(f"Still incomplete ({final_size/1e6:.1f} / ~371 MB) — rerun this script to keep resuming.")
