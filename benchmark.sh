#!/bin/bash
#SBATCH -o benchmark_time.%j.out      # 输出文件名
#SBATCH --partition=l40s              # 使用你可用的分区
#SBATCH --qos=dcgpu                   # 使用你可用的QOS
#SBATCH -J time-compare               # 作业名
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # 我们只需要1个GPU
#SBATCH --cpus-per-task=8

# --- 确保你的conda环境被激活 ---
# 如果需要，取消下面这行的注释并替换成你的conda环境路径
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate llava

# --- 打印环境信息用于调试 ---
echo "========= 环境信息 ========="
echo "作业运行节点: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python 路径: $(which python)"
echo "PyTorch 版本: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi
echo "============================="

# --- 运行我们的基准测试脚本 ---
echo "正在启动时间对比测试脚本..."
python benchmark_latency.py
echo "测试脚本执行结束。"
