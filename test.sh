#!/bin/bash
#SBATCH -o llava.%j.out               # 脚本执行的输出将被保存在 llava.%j.out文件下，%j表示作业号
#SBATCH --partition=l40s
#SBATCH --qos=dcgpu                    # 指定作业的 QOS
#SBATCH -J llava                      # 作业在调度系统中的作业名为 llava
#SBATCH --nodes=1                     # 申请节点数为 1
#SBATCH --ntasks-per-node=1           # 每个节点上运行的任务数为 4
#SBATCH --gres=gpu:1                  # 指定作业需要的 GPU卡数量为 4
#SBATCH --cpus-per-task=8             # 每个任务分配的 CPU核心数
# SBATCH --nodelist=l40sgpu001        # 指定作业只能在 node1和 node2节点上执行
# a100gpu003,l40gpu001

# ==== 设置环境变量 ====
export PYTHONPATH="/lab/zhangjg_lab/30028000/llava:$PYTHONPATH"
export PATH="/lab/zhangjg_lab/30028000/.conda/envs/llava/bin:$PATH"
nvidia-smi
# ==== 调试信息 ====
echo "Python 路径: $(which python)"
echo "accelerate 路径: $(which accelerate)"

python generate_coco_caption.py
