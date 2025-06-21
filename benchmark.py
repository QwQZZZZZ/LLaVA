import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- 1. 配置 ---
# MODEL_PATH 可以是 HuggingFace Hub 名称（如 "state-spaces/mamba-2.8b-hf"）或本地transformers格式权重目录
MODEL_PATH = "model/falcon-mamba-7b"  # <--- 可替换为本地transformers模型路径
SEQUENCE_LENGTHS = [
    16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640, 768,
    896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096
]
WARMUP_RUNS = 3
TIMED_RUNS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_mamba_benchmark_with_transformers():
    print("使用设备:", DEVICE)
    print("Mamba Transformers 模型路径:", MODEL_PATH)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"./mamba_transformers_benchmark_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 加载 tokenizer
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
            tokenizer.add_special_tokens({'pad_token': tokenizer.pad_token})
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer 词汇表大小: {vocab_size}")
    except Exception as e:
        print(f"警告: 无法从 '{MODEL_PATH}' 加载 tokenizer。请确保路径正确且包含tokenizer文件。错误: {e}")
        print("将使用一个默认的词汇表大小 (50257) 来生成随机输入ID。")
        tokenizer = None
        vocab_size = 50257

    # 加载 Transformers 版本的 Mamba 模型
    try:
        # 注意：如需float16请确保GPU支持
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
    except Exception as e:
        print(f"错误: 无法从 '{MODEL_PATH}' 加载 Transformers Mamba 模型。")
        print(f"请检查路径、模型完整性或transformers版本。")
        print(f"具体错误: {e}")
        return

    model.eval()

    # 获取模型最大可处理序列长度
    # 一般 transformers config 有 max_position_embeddings
    try:
        max_seq_len = getattr(model.config, "max_position_embeddings", max(SEQUENCE_LENGTHS))
    except Exception:
        max_seq_len = max(SEQUENCE_LENGTHS)

    print("模型支持的最大序列长度:", max_seq_len)

    results = []
    for seq_len in SEQUENCE_LENGTHS:
        if seq_len > max_seq_len:
            print(f"跳过序列长度 {seq_len}，超过模型最大支持长度 {max_seq_len}。")
            continue

        # 生成随机输入 ID
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)

        # 预热
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = model(input_ids)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        # 正式计时
        times = []
        with torch.no_grad():
            for _ in range(TIMED_RUNS):
                start = time.time()
                _ = model(input_ids)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"[{seq_len}] 平均前向传播延迟: {avg_time * 1000:.2f} ms")

        results.append({
            "sequence_length": seq_len,
            "latency_ms": avg_time * 1000,
        })

    # 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "mamba_transformers_forward_latency.csv")
    df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(df["sequence_length"], df["latency_ms"], marker='o', linestyle='-')
    plt.title(f"Mamba Model (Transformers) Forward Pass Latency vs Sequence Length ({os.path.basename(MODEL_PATH)})")
    plt.xlabel("Input Sequence Length")
    plt.ylabel("Average Forward Pass Latency (ms)")
    plt.grid(True)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "mamba_transformers_forward_latency_plot.png")
    plt.savefig(png_path)
    print(f"图像已保存到: {png_path}")

if __name__ == "__main__":
    run_mamba_benchmark_with_transformers()