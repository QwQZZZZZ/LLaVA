import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import traceback  # 导入 traceback 用于更详细的错误报告

from transformers import AutoTokenizer, AutoModelForCausalLM

# 确保 LlavaLlamaForCausalLM 在你的环境中可导入
# 如果你的llava安装在特定位置，需要确保python路径正确
try:
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
except ImportError:
    print("警告: 无法从 'llava.model.language_model.llava_llama' 导入 LlavaLlamaForCausalLM。")
    print("请确保 LLaVA 模块已正确安装，且其路径已添加到 PYTHONPATH。")
    # 如果无法导入，将无法测试 LLaVA 模型
    LlavaLlamaForCausalLM = None

# --- 1. 配置 ---
MODELS_TO_TEST = {
    "LLaVA-v1.5-7B (Llama-2-base)": {
        "model_id": "/lab/zhangjg_lab/30028000/llava/models/llava-v1.5-7b",
        "model_class": "LlavaLlamaForCausalLM",  # 明确指定类名
        "trust_remote_code": False  # 本地加载通常不需要
    },
    "Qwen-7B": {
        "model_id": "Qwen/Qwen-7B",
        "model_class": "AutoModelForCausalLM",  # 使用AutoModelForCausalLM加载Qwen
        "trust_remote_code": True  # Qwen模型通常需要
    }
}

# 确保序列长度适用于所有模型，Llama和Qwen-7B的上下文窗口都是4096或8192
SEQUENCE_LENGTHS = [
    128, 256, 384, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096
]
WARMUP_RUNS = 3
TIMED_RUNS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. 主逻辑 ---
def run_benchmark():
    print("开始运行模型延迟基准测试...")
    print(f"使用设备: {DEVICE}")
    if DEVICE == "cpu":
        print("警告: 未检测到 CUDA，将在 CPU 上运行。速度会非常慢！")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"./benchmark_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有结果将保存在: {output_dir}\n")

    all_results_data = []

    for model_name, config in MODELS_TO_TEST.items():
        print("=" * 30)
        print(f"正在测试模型: {model_name} ({config['model_id']})")
        print("=" * 30)

        tokenizer = None
        model = None
        model_max_len = float('inf')  # Default large value

        try:
            print(f" -> 正在加载 {model_name} 的 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                config['model_id'], trust_remote_code=config.get("trust_remote_code", False)
            )

            if "Qwen" in model_name:
                # Qwen tokenizer 不允许添加任意的特殊token。
                # 它通常使用其 eos_token 作为 pad_token。
                # 我们直接赋值，不再尝试通过 add_special_tokens 添加。
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    # 打印警告，但不要尝试添加，因为Qwen的tokenizer会拒绝
                    print(f"警告: {model_name} 的pad_token未设置，已设为其eos_token '{tokenizer.pad_token}'。")
            else:  # 对于其他模型，例如LLaVA/Llama
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
                    if tokenizer.pad_token == "[PAD]":
                        # 只有当 fallback 到自定义的 [PAD] 时才需要 add_special_tokens
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print(f"警告: {model_name} 的pad_token未明确设置，已设为 '{tokenizer.pad_token}'。")

            print(f" -> 正在加载 {model_name} 到 {DEVICE}...")
            if config["model_class"] == "LlavaLlamaForCausalLM":
                if LlavaLlamaForCausalLM is None:
                    raise ImportError("LlavaLlamaForCausalLM 类未成功导入，无法加载 LLaVA 模型。")
                model = LlavaLlamaForCausalLM.from_pretrained(
                    config['model_id'],
                    torch_dtype=torch.float16
                ).to(DEVICE)
            elif config["model_class"] == "AutoModelForCausalLM":
                model = AutoModelForCausalLM.from_pretrained(
                    config['model_id'],
                    torch_dtype=torch.float16,
                    trust_remote_code=config.get("trust_remote_code", False)
                ).to(DEVICE)
            else:
                raise ValueError(f"无法识别的模型类: {config['model_class']}")

            model.eval()  # 设置为评估模式

            model_max_len = getattr(model.config, 'max_position_embeddings', 4096)
            print(f" -> 模型加载成功。检测到最大序列长度: {model_max_len}")

        except Exception as e:
            print(f"\n加载模型 {config['model_id']} 失败: {e}\n")
            traceback.print_exc()  # 打印完整的错误栈信息
            continue  # 跳过当前模型，测试下一个

        model_results = []
        for seq_len in SEQUENCE_LENGTHS:
            if seq_len > model_max_len:
                print(f" -> 跳过序列长度 {seq_len}，超过模型最大支持长度 ({model_max_len})。")
                continue

            print(f"\n--- 测试模型: {model_name}, 序列长度: {seq_len} ---")
            # 确保 input_ids 的长度不会超过 tokenizer 的模型最大长度限制
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=DEVICE)

            print(" -> 正在进行预热运行...")
            with torch.no_grad():
                for _ in range(WARMUP_RUNS):
                    _ = model.generate(input_ids, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id,
                                       do_sample=False)
            torch.cuda.synchronize()  # 确保GPU操作完成

            print(f" -> 正在进行 {TIMED_RUNS} 次计时运行...")
            timings = []
            with torch.no_grad():
                for _ in range(TIMED_RUNS):
                    start_time = time.time()
                    _ = model.generate(input_ids, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id,
                                       do_sample=False)
                    torch.cuda.synchronize()  # 确保GPU操作完成
                    end_time = time.time()
                    timings.append(end_time - start_time)

            avg_time_s = sum(timings) / len(timings)
            latency_ms = avg_time_s * 1000
            print(f" -> 平均延迟: {latency_ms:.2f} ms")

            model_results.append({
                "model_name": model_name,
                "sequence_length": seq_len,
                "average_time_s": avg_time_s,
                "latency_ms": latency_ms
            })
        all_results_data.extend(model_results)  # 将当前模型的结果添加到总列表中

    if not all_results_data:
        print("\n没有收集到任何数据，无法生成报告。")
        return

    print("\n\n基准测试完成。正在生成报告...")
    df = pd.DataFrame(all_results_data)
    csv_path = os.path.join(output_dir, "benchmark_results_combined.csv")
    df.to_csv(csv_path, index=False)
    print(f" -> 详细数据已保存到: {csv_path}")

    print("\n--- 实验数据总结 ---")
    print(df.to_string())

    # --- 绘图部分 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制所有模型的数据
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        ax.plot(model_df['sequence_length'], model_df['latency_ms'], marker='o', label=model_name)

    # 统一设置 X 和 Y 轴
    ax.set_title('Model Inference Latency vs. Sequence Length', fontsize=16)
    ax.set_xlabel('Input Sequence Length', fontsize=14)
    ax.set_ylabel('Average Latency (ms)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)

    # 确保 X 轴刻度包含所有测试长度
    ax.set_xticks(SEQUENCE_LENGTHS)
    ax.tick_params(axis='x', rotation=45)  # 旋转X轴标签，防止重叠

    # 自动调整 Y 轴范围，以清晰显示数据
    min_latency = df['latency_ms'].min() * 0.9  # 留一点空白
    max_latency = df['latency_ms'].max() * 1.1  # 留一点空白
    ax.set_ylim(min_latency, max_latency)

    # 可以选择性地使用对数尺度，但为了看线性/二次趋势，通常不建议Y轴用对数
    # 如果数据范围太大，考虑Y轴用对数，或者分段绘图
    # ax.set_yscale('log')

    plt.tight_layout()  # 自动调整布局，防止标签重叠
    plot_path = os.path.join(output_dir, "latency_vs_seqlen_comparison.png")
    plt.savefig(plot_path)
    print(f" -> 对比图表已保存到: {plot_path}")
    print("\n脚本执行完毕。")


if __name__ == "__main__":
    run_benchmark()
