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
#export PYTHONPATH="/lab/zhangjg_lab/30028000/llava:$PYTHONPATH"
#export PATH="/lab/zhangjg_lab/30028000/.conda/envs/llava/bin:$PATH"
nvidia-smi
# ==== 调试信息 ====
echo "Python 路径: $(which python)"
echo "accelerate 路径: $(which accelerate)"


# --- LLaVA 训练参数配置 ---

# 注意：这些路径是基于您 LLaVA 项目的根目录 (LLaVA/)。
# 确保您在运行此脚本时，当前工作目录是 LLaVA 项目的根目录。

# --- 模型选择 ---
# 语言模型: Vicuna-7B v1.5 (教程推荐的最小模型)
# 如果您已将模型下载到本地，请修改为本地路径，例如：
# MODEL_BASE="/path/to/your/models/vicuna-7b-v1.5"
MODEL_BASE="/lab/zhangjg_lab/30028000/llava/models/llava-v1.5-7b"

# 视觉编码器: CLIP ViT-Base Patch32 (教程推荐的最小视觉编码器)
# 如果您已将模型下载到本地，请修改为本地路径，例如：
# VISION_TOWER="/path/to/your/models/clip-vit-base-patch32"
#VISION_TOWER="/lab/zhangjg_lab/30028000/llava/models/resnet-50"
VISION_TOWER="resnet50"

# --- 数据路径 ---
# 您上一步生成的COCO Captions训练数据JSON文件
DATA_PATH="/lab/zhangjg_lab/30028000/llava/data/coco/annotations/llava_coco_captions_train.json"
# COCO图片所在的根目录 (包含 train2014, val2014 或 train2017 等文件夹的父目录)
# 请务必修改为您的实际路径，并确保是绝对路径或相对于您执行脚本的路径
IMAGE_FOLDER="/lab/zhangjg_lab/30028000/llava/data/coco" # 示例：假设您下载到了这个路径

# --- 输出路径 ---
OUTPUT_DIR="./checkpoints/llava-v1.5-resnet50-lora-coco-caption-test"

# --- DeepSpeed 配置 ---
# 根据教程，这里是zero3.json，而不是您原始脚本的zero2.json
# 确保您LLaVA项目根目录下的 scripts/zero3.json 文件存在且配置正确
DEEPSPEED_CONFIG="./scripts/zero3.json"

# --- 训练参数 ---
# 如果您的GPU显存不足(如24G)，可以减小per_device_train_batch_size，同时增大gradient_accumulation_steps
# 保证 per_device_train_batch_size * gradient_accumulation_steps * num_gpus (全局批次大小) 大致不变
# 教程建议 per_device_train_batch_size=4, gradient_accumulation_steps=4
# 对于单张A100 (40G/80G)，这个配置通常是可行的。
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4

# LoRA 参数
LORA_R=128
LORA_ALPHA=256
MM_PROJECTOR_LR="2e-5" # 注意：这个参数在原始 LLaVA 脚本中没有

echo "开始 LLaVA LoRA 微调训练..."
echo "LLM Base Model: $MODEL_BASE"
echo "Vision Tower: $VISION_TOWER"
echo "Data Path: $DATA_PATH"
echo "Image Folder: $IMAGE_FOLDER"
echo "Output Directory: $OUTPUT_DIR"
echo "Deepspeed Config: $DEEPSPEED_CONFIG"
echo "Per Device Train Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"

deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_BASE \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo "LLaVA LoRA 微调训练结束。"
