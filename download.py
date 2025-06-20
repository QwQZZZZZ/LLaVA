from huggingface_hub import snapshot_download

local_dir = "/lab/zhangjg_lab/30028000/llava/models/qwen-7b" # 指定下载到哪个本地文件夹
snapshot_download(repo_id="Qwen/Qwen-7B", local_dir=local_dir, local_dir_use_symlinks=False)
print(f"模型已下载到: {local_dir}")
