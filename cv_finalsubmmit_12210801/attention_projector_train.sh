# 模型和数据路径
MODEL_BASE="/home/moting/llava_project/models/llava-v1.5-7b"
VISION_TOWER="/home/moting/llava_project/models/clip-vit-large-patch14-336"
DATA_PATH="/home/moting/llava_project/coco2014/annotations/annotations/llava_coco_captions_train.json"
IMAGE_FOLDER="/home/moting/llava_project/coco2014"

# 输出目录：会自动创建一个以当前时间命名的文件夹
OUTPUT_DIR="./checkpoints/llava-lora-attention-projector_$(date +%Y%m%d_%H%M%S)"

# 训练配置
PROJECTOR_TYPE="attention"      # <-- 指定我们新的 Attention Projector
DEEPSPEED_CONFIG="/home/moting/llava_project/LLaVA/scripts/zero3.json" # DeepSpeed 配置文件路径
NUM_TRAIN_EPOCHS=1              # 训练轮次
PER_DEVICE_TRAIN_BATCH_SIZE=16   # 根据你的显存调整，Attention Projector 可能比 MLP 更耗显存
GRADIENT_ACCUMULATION_STEPS=4   # 梯度累积步数

# LoRA 参数
LORA_R=128
LORA_ALPHA=256

# !! 关键：学习率配置 !!
# 我们使用一个同步且稳定的学习率，防止训练崩溃。
LEARNING_RATE="2e-4"      # LoRA 模块的学习率
PROJECTOR_LR="2e-5"     # Projector 模块的学习率


# --- 3. 预运行检查和日志 ---
echo "==========================================================="
echo "启动 LLaVA 微调任务 (Attention Projector)"
echo "==========================================================="
echo "GPU(s) being used: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "---"
echo "模型配置:"
echo "  - 基础模型 (LLM): $MODEL_BASE"
echo "  - 视觉塔 (Vision Tower): $VISION_TOWER"
echo "  - Projector 类型: $PROJECTOR_TYPE"
echo "---"
echo "数据配置:"
echo "  - 标注文件: $DATA_PATH"
echo "  - 图像目录: $IMAGE_FOLDER"
echo "---"
echo "训练参数:"
echo "  - 学习率 (LoRA): $LEARNING_RATE"
echo "  - 学习率 (Projector): $PROJECTOR_LR"
echo "  - 训练轮次: $NUM_TRAIN_EPOCHS"
echo "  - 设备批大小: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  - 梯度累积: $GRADIENT_ACCUMULATION_STEPS"
echo "---"
echo "输出目录:"
echo "  - $OUTPUT_DIR"
echo "==========================================================="

# 确认 GPU 状态
nvidia-smi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"


# --- 4. 核心训练命令 ---
# 启动 DeepSpeed 进行分布式训练
deepspeed --include localhost:2,3,5,6,7 /home/moting/llava_project/LLaVA/llava/train/train_mem.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL_BASE" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type "$PROJECTOR_TYPE" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 120 \
    --save_total_limit 1 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --lora_weight_path "" \
    --lora_bias "none" \
    --mm_projector_lr "$PROJECTOR_LR" \
    --freeze_vision_tower True # !! 关键 !! 确保视觉塔被冻结

# --- 5. 结束 ---
echo "==========================================================="
echo "训练任务已结束。"
echo "最终模型检查点保存在: $OUTPUT_DIR"
echo "==========================================================="