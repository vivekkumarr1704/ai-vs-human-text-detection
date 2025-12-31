from huggingface_hub import snapshot_download

print("Downloading HC3 dataset...")

snapshot_download(
    repo_id="Hello-SimpleAI/HC3",
    repo_type="dataset",
    local_dir="hc3_raw",
    local_dir_use_symlinks=False
)

print("✅ HC3 dataset downloaded successfully!")
