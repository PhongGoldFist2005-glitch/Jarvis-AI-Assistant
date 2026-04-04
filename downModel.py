from huggingface_hub import snapshot_download

# # Tải toàn bộ model về thư mục ./PhoWhisper-tiny
snapshot_download(
    repo_id="vinai/PhoWhisper-tiny",
    local_dir="PhoWhisper-tiny",
    local_dir_use_symlinks=False
)