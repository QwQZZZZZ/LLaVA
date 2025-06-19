import argparse
import torch
import json
import os
from tqdm import tqdm

# 从 LLaVA 库导入必要的组件
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path # KeywordsStoppingCriteria 在批量模式下通常不直接使用

from PIL import Image

# 导入用于评估的库
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def generate_captions(args):
    """
    使用指定的 LLaVA 模型为 COCO 验证集中的图像生成描述。
    批量处理以提高效率。
    """
    # 禁用 PyTorch 初始化，避免可能的警告
    disable_torch_init()

    print("--- 正在加载模型 ---")
    model_name = get_model_name_from_path(args.model_path)
    # 加载基座模型和 LoRA 权重
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    # 设置 tokenizer 的 padding token，这对于批量处理非常重要
    # 如果没有设置，通常使用 eos_token 作为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # 根据模型名称确定对话模式
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # 处理用户指定的对话模式，如果与自动识别的不同则发出警告
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f'[警告] 自动确定的对话模式是 {conv_mode}，而 `--conv-mode` 设置为 {args.conv_mode}。将使用 {args.conv_mode}。')
        conv_mode = args.conv_mode
    else:
        args.conv_mode = conv_mode

    print("--- 正在准备验证集数据 ---")
    with open(args.coco_val_annotations_file, 'r') as f:
        val_data = json.load(f)

    val_images = {img['id']: img for img in val_data['images']}
    image_ids = list(val_images.keys())

    results = [] # 存储生成的描述

    print(f"--- 正在生成描述，批处理大小: {args.batch_size} ---")

    # --- 批量处理逻辑 ---
    batch_image_ids = []      # 存储当前批次的图像ID
    batch_image_tensors = []  # 存储当前批次的图像张量
    batch_input_ids = []      # 存储当前批次的输入ID张量

    # 准备固定的提示部分，提高循环效率
    qs = "<image>\nPlease generate a caption for this image."
    conv_template_base = conv_templates[args.conv_mode]

    for i, image_id in enumerate(tqdm(image_ids, desc="生成描述")):
        image_info = val_images[image_id]
        image_file = os.path.join(args.image_folder, image_info['file_name'])

        if not os.path.exists(image_file):
            print(f"跳过缺失的图像文件: {image_file}")
            continue

        # --- 为当前图像构建输入 ---
        # 复制对话模板以确保每个图像都从新的对话开始，避免状态混淆
        conv = conv_template_base.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None) # None 表示模型将填充此处的回复
        prompt = conv.get_prompt() # 获取完整的提示字符串

        try:
            image = Image.open(image_file).convert('RGB')
            # 图像预处理，并移动到半精度（FP16）
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half()
            # 将提示字符串转换为 token ID，并插入图像 token
            input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').squeeze(0)

            # 将当前图像的相关信息添加到批次列表中
            batch_image_ids.append(image_id)
            batch_image_tensors.append(image_tensor)
            batch_input_ids.append(input_id)

        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {e}")
            continue

        # 如果批次已满，或者已经是最后一张图像且批次中还有内容，则处理当前批次
        if (len(batch_image_ids) == args.batch_size) or (i == len(image_ids) - 1 and len(batch_image_ids) > 0):
            # 对输入 ID 进行填充：找到当前批次中最长的序列，并将其余序列填充到相同长度
            max_len = max(len(ids) for ids in batch_input_ids)
            padded_input_ids = torch.stack([
                torch.cat([ids, torch.full((max_len - len(ids),), tokenizer.pad_token_id, dtype=ids.dtype)])
                for ids in batch_input_ids
            ]).to(args.device) # 将填充后的输入ID移动到指定设备

            # 堆叠图像张量，形成批处理维度
            stacked_image_tensors = torch.cat(batch_image_tensors).to(args.device)

            # 获取停止字符串（用于后处理去除模型生成的多余部分）
            stop_str = conv_template_base.sep if conv_template_base.sep_style != SeparatorStyle.TWO else conv_template_base.sep2
            # 注意：KeywordsStoppingCriteria 通常不适用于批量推理，因为每个序列停止点可能不同。
            # 这里我们依赖 max_new_tokens 并进行后处理。

            with torch.inference_mode(): # 禁用梯度计算，节省内存并加速
                output_ids_batch = model.generate(
                    padded_input_ids,
                    images=stacked_image_tensors,
                    do_sample=False,  # 使用 Greedy search 保证结果一致性
                    temperature=0,    # Greedy search 下温度设为0
                    max_new_tokens=1024, # 最大生成 token 数
                    use_cache=True,   # 启用 KV 缓存加速生成
                    stopping_criteria=None # 批量推理时通常不使用复杂的 stopping_criteria
                )

            # 解码批次中的每个输出
            for j in range(output_ids_batch.shape[0]):
                # 获取当前批次中第 j 个序列的原始输入长度
                original_input_len = len(batch_input_ids[j])
                # 解码生成的 token，去除输入提示部分
                outputs = tokenizer.decode(output_ids_batch[j, original_input_len:]).strip()

                # 清理输出：如果以停止字符串结尾，则去除
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)].strip()

                results.append({"image_id": batch_image_ids[j], "caption": outputs})

            # 清空批次列表，为下一个批次做准备
            batch_image_ids = []
            batch_image_tensors = []
            batch_input_ids = []

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- 正在保存生成的描述到 {args.output_file} ---")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"所有描述已生成并保存到 {args.output_file}")


def evaluate_captions(ground_truth_file, result_file):
    """
    将生成的描述与 COCO 地面真值注释进行评估。
    """
    print("\n--- 正在开始评估 ---")
    try:
        coco = COCO(ground_truth_file)
        coco_result = coco.loadRes(result_file)

        coco_eval = COCOEvalCap(coco, coco_result)

        # 只评估有对应结果的图片
        coco_eval.params['image_id'] = coco_result.getImgIds()

        coco_eval.evaluate()

        print("\n--- 评估分数 ---")
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.4f}')  # 格式化分数，保留4位小数
        print("-------------------------")
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        print("请确保地面真值文件和结果文件格式正确且可访问。")


def main():
    parser = argparse.ArgumentParser(description="生成图像描述并对照 COCO 数据集进行评估。")
    parser.add_argument("--model-path", type=str, default='./checkpoints/llava-v1.5-7b-lora-coco-caption',
                        help="LoRA 权重路径。")
    parser.add_argument("--model-base", type=str, default='/lab/zhangjg_lab/30028000/llava/models/llava-v1.5-7b',
                        help="基座模型路径。")
    parser.add_argument("--coco-val-annotations-file", type=str,
                        default="/lab/zhangjg_lab/30028000/llava/data/coco/annotations/captions_val2014.json",
                        help="COCO 验证集标注文件（地面真值）路径。")
    parser.add_argument("--image-folder", type=str, default="/lab/zhangjg_lab/30028000/llava/data/coco/val2014",
                        help="包含 COCO 验证集图像的文件夹路径。")
    parser.add_argument("--output-file", type=str, default="./result/coco_val_results.json",
                        help="保存生成描述的路径。")
    parser.add_argument("--device", type=str, default="cuda",
                        help="用于推理的设备（例如：'cuda' 或 'cpu'）。")
    parser.add_argument("--conv-mode", type=str, default=None,
                        help="LLaVA 的对话模式（例如：'llava_v1', 'llava_llama_2'）。")
    parser.add_argument("--load-8bit", action="store_true",
                        help="以 8 位量化加载模型。")
    parser.add_argument("--load-4bit", action="store_true",
                        help="以 4 位量化加载模型。")
    parser.add_argument("--batch-size", type=int, default=16, # 增加默认批处理大小，可以根据显存调整
                        help="一次处理的图像数量。增加此值可以加快推理速度。")
    args = parser.parse_args()

    # 扩展用户主目录路径，确保路径正确解析
    args.coco_val_annotations_file = os.path.expanduser(args.coco_val_annotations_file)
    args.image_folder = os.path.expanduser(args.image_folder)
    args.output_file = os.path.expanduser(args.output_file) # 确保输出文件路径也经过扩展

    # 步骤 1: 生成描述
    generate_captions(args)

    # 步骤 2: 评估描述
    # 使用与生成结果相同的文件进行评估
    evaluate_captions(args.coco_val_annotations_file, args.output_file)


if __name__ == "__main__":
    main()
