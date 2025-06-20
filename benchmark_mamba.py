import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 从 mamba_ssm 库导入 Mamba 模型类
from mamba_ssm.models.mixer_mamba import MambaLMHeadModel
# 导入 Hugging Face Transformers 的 AutoTokenizer，Mamba 模型通常仍然需要标准的 tokenizer
from transformers import AutoTokenizer

# --- 1. 配置 ---
# 请根据你的实际情况修改以下路径
# MAMBA_MODEL_PATH 应该指向包含模型权重和配置文件的文件夹
MAMBA_MODEL_PATH = "/path/to/your/local/mamba-model" # <--- **请务必替换为你的本地 Mamba 模型路径**
SEQUENCE_LENGTHS = [
    128, 256, 384, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096
]
WARMUP_RUNS = 3 # 与LLaVA脚本保持一致
TIMED_RUNS = 10 # 与LLaVA脚本保持一致
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 主逻辑 ---
def run_mamba_benchmark_with_mamba_ssm():
    print("使用设备:", DEVICE)
    print("Mamba 模型路径:", MAMBA_MODEL_PATH)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"./mamba_ssm_benchmark_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MAMBA_MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
            tokenizer.add_special_tokens({'pad_token': tokenizer.pad_token})
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer 词汇表大小: {vocab_size}")
    except Exception as e:
        print(f"警告: 无法从 '{MAMBA_MODEL_PATH}' 加载 tokenizer。请确保路径正确且包含tokenizer文件。错误: {e}")
        print("将使用一个默认的词汇表大小 (50257) 来生成随机输入ID。")
        tokenizer = None
        vocab_size = 50257 # 默认词汇表大小

    # 加载 Mamba 模型
    try:
        # mamba_ssm 库的 MambaLMHeadModel.from_pretrained 方法
        # 需要传入配置，通常从 config.json 中读取
        # 或者直接传入参数，例如 d_model, n_layer 等
        # 这里的例子假设模型路径包含了 transformers 兼容的 config.json
        model = MambaLMHeadModel.from_pretrained(MAMBA_MODEL_PATH, device=DEVICE, dtype=torch.float16)
    except Exception as e:
        print(f"错误: 无法从 '{MAMBA_MODEL_PATH}' 加载 Mamba 模型。请检查路径和模型完整性。")
        print(f"使用 mamba_ssm.models.mixer_mamba.MambaLMHeadModel 加载失败。")
        print(f"请确保模型路径包含适用于 mamba_ssm 的正确配置文件和权重文件。")
        print(f"具体错误: {e}")
        return # 退出程序

    model.eval() # 设置为评估模式，与LLaVA脚本一致

    # 获取模型最大序列长度
    # mamba_ssm 模型通常在 config 中有 d_model, n_layer 等参数，但没有直接的 max_position_embeddings
    # Mamba 理论上支持任意长序列，但为了和LLaVA对比，我们保持相同的测试序列范围
    # 如果Mamba模型有配置 max_length 或 context_length，请自行调整
    max_seq_len = max(SEQUENCE_LENGTHS) # 默认为测试列表中的最大值，因为Mamba理论上可以处理很长的序列
    print("模型测试的最大序列长度:", max_seq_len)

    results = []
    for seq_len in SEQUENCE_LENGTHS:
        # 在Mamba中通常不需要跳过，但为了保持逻辑一致性
        if seq_len > max_seq_len:
            print(f"跳过序列长度 {seq_len}，超过模型测试的最大长度 {max_seq_len}。")
            continue

        # 生成随机输入 ID
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=DEVICE)

        # 预热 (单次前向传播)
        with torch.no_grad():
            for _ in range(WARMUP_RUNS):
                # MambaLMHeadModel 的 forward 方法接受 input_ids
                # 它返回 logits，我们只关心计算时间
                _ = model(input_ids)
        torch.cuda.synchronize() # 确保所有CUDA操作完成

        # 计时 (单次前向传播)
        times = []
        with torch.no_grad():
            for _ in range(TIMED_RUNS):
                start = time.time()
                _ = model(input_ids) # 衡量Mamba核心的前向传播延迟
                torch.cuda.synchronize() # 确保所有CUDA操作完成
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
    csv_path = os.path.join(output_dir, "mamba_ssm_forward_latency.csv")
    df.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(df["sequence_length"], df["latency_ms"], marker='o', linestyle='-')
    plt.title(f"Mamba Model (mamba_ssm) Forward Pass Latency vs Sequence Length ({os.path.basename(MAMBA_MODEL_PATH)})")
    plt.xlabel("Input Sequence Length")
    plt.ylabel("Average Forward Pass Latency (ms)")
    plt.grid(True)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "mamba_ssm_forward_latency_plot.png")
    plt.savefig(png_path)
    print(f"图像已保存到: {png_path}")

if __name__ == "__main__":
    run_mamba_benchmark_with_mamba_ssm()
