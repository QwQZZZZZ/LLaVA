import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from transformers import AutoTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

# --- 1. 配置 ---
MODEL_PATH = "/lab/zhangjg_lab/30028000/llava/models/llava-v1.5-7b"
SEQUENCE_LENGTHS = [
    128, 256, 384, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096
]
WARMUP_RUNS = 3
TIMED_RUNS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 主逻辑 ---
def run_benchmark():
    print("使用设备:", DEVICE)
    print("模型路径:", MODEL_PATH)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"./benchmark_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
        tokenizer.add_special_tokens({'pad_token': tokenizer.pad_token})

    model = LlavaLlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    # 获取最大序列长度
    max_seq_len = getattr(model.config, "max_position_embeddings", 4096)
    print("最大序列长度:", max_seq_len)

    results = []
    for seq_len in SEQUENCE_LENGTHS:
        if seq_len > max_seq_len:
            print(f"跳过序列长度 {seq_len}，超过模型支持的最大长度。")
            continue

        input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=DEVICE)

        # 预热
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                _ = model.generate(input_ids, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()

        # 计时
        times = []
        with torch.no_grad():
            for _ in range(TIMED_RUNS):
                start = time.time()
                _ = model.generate(input_ids, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"[{seq_len}] 平均延迟: {avg_time * 1000:.2f} ms")

        results.append({
            "sequence_length": seq_len,
            "latency_ms": avg_time * 1000,
        })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "llava_latency.csv"), index=False)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(df["sequence_length"], df["latency_ms"], marker='o')
    plt.title("LLaVA-v1.5-7B Latency vs Sequence Length")
    plt.xlabel("Input Sequence Length")
    plt.ylabel("Average Latency (ms)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "llava_latency_plot.png"))
    print(f"图像已保存到: {output_dir}")

if __name__ == "__main__":
    run_benchmark()
