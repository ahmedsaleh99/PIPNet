import os
from huggingface_hub import snapshot_download

# Your Hugging Face repo ID
REPO_ID = "assale02/PIPNet"

# Where to download the repo
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

def download_repo():
    if not os.path.exists(ROOT_DIR) or not os.listdir(ROOT_DIR):
        print(f"Downloading entire Hugging Face repo {REPO_ID} → {ROOT_DIR}")
        snapshot_download(repo_id=REPO_ID, local_dir=ROOT_DIR, allow_patterns=["*"])
        print("Download complete.")
    else:
        print(f"{ROOT_DIR} already exists and is not empty, skipping download.")

if __name__ == "__main__":
    download_repo()

