import argparse
import torch
from datetime import datetime
import json
import os
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model  # 确保这个函数能够正确加载LLaVA模型
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image


def main(args):
    # 禁用 PyTorch 初始化时的冗余信息，避免干扰
    disable_torch_init()

    # --- 加载模型 ---
    # 根据 llava.model.builder.py 的设计，
    # 如果 model_base 为 None 且 model_path 是一个完整的 LLaVA 模型，
    # 则 load_pretrained_model 会直接加载完整的 LLaVA 模型。
    # 这里我们设定 model_path 为原始的 LLaVA-v1.5-7b 模型。
    model_name = get_model_name_from_path(args.model_path)
    print(f"正在从 {args.model_path} 加载模型...")
    print(f"模型名称推断为: {model_name}")

    # load_pretrained_model 函数会自动处理模型加载，包括可能的LoRA融合
    # 如果 args.model_path 是完整的 LLaVA 模型，且 args.model_base 为 None，
    # 那么它会直接加载该完整的 LLaVA 模型。
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base if args.model_base else None,  # 如果未提供 model_base，则传入 None
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device
    )
    print("模型加载完成。")

    # --- 确定对话模式 ---
    # 根据模型名称自动选择对话模板
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():  # 对应 LLaVA v1.5 模型
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"  # 默认模式

    # 如果命令行指定了 conv_mode，则覆盖自动判断的结果
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            f'[警告] 自动确定的对话模式是 {conv_mode}，而命令行指定了 --conv-mode={args.conv_mode}，将使用 {args.conv_mode}。')
        conv_mode = args.conv_mode
    else:
        args.conv_mode = conv_mode  # 将最终使用的 conv_mode 存回 args

    # --- 准备验证集数据 ---
    print(f"正在加载 COCO 验证集标注文件: {args.coco_val_annotations_file}")
    with open(args.coco_val_annotations_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # 构建图像信息字典，方便通过 image_id 查找
    val_images = {img['id']: img for img in val_data['images']}
    # 获取所有图像的 ID 列表
    all_image_ids = list(val_images.keys())

    # --- 选取前 10 张图片进行测试 ---
    # 确保 image_ids 是一个可迭代对象，并包含实际的图像 ID
    # 这里取前10个 ID，如果 all_image_ids 不足10个，就取全部
    test_image_ids = all_image_ids[:args.num_test_images]
    print(f"将测试前 {len(test_image_ids)} 张图片。")

    results = []  # 存储生成的描述

    # --- 循环生成描述 ---
    for image_id in tqdm(test_image_ids, desc="生成图片描述"):
        image_info = val_images[image_id]
        # 拼接完整的图像文件路径
        image_file = os.path.join(args.image_folder, image_info['file_name'])

        # 检查图像文件是否存在
        if not os.path.exists(image_file):
            print(f"警告：图像文件 {image_file} 不存在，跳过。")
            continue

        # --- 构建模型输入 ---
        # 针对图像描述任务的提示语
        qs = "<image>\n请为这张图片生成一个描述。"  # 示例中文提示
        # qs = "<image>\nPlease generate a caption for this image." # 英文提示

        # 复制对话模板，并添加用户消息和模型占位符
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)  # 用户角色说出提示语
        conv.append_message(conv.roles[1], None)  # 模型角色等待回答
        prompt = conv.get_prompt()  # 获取完整的提示字符串

        # 加载并预处理图像
        image = Image.open(image_file).convert('RGB')
        # image_processor.preprocess 返回一个字典，包含 'pixel_values'
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)

        # 标记化提示字符串，并将图像 token 索引替换为实际的图像张量
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            args.device)

        # 设置停止生成条件，当模型生成到特定分隔符时停止
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # --- 模型生成 ---
        with torch.inference_mode():  # 禁用梯度计算，节省显存并加速推理
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,  # 使用 Greedy search 保证结果确定性
                temperature=0,  # Greedy search 下 temperature 设为 0
                max_new_tokens=1024,  # 最大生成 token 数量
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # 解码生成的 token
        # output_ids[0, input_ids.shape[1]:] 表示跳过输入提示部分，只解码生成的响应
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        # 清理输出：移除停止字符串
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        # 将结果添加到列表中
        results.append({"image_id": image_id, "caption": outputs})

    # --- 保存结果 ---
    print(f"所有描述已生成，准备保存到 {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        # 使用 indent=2 和 ensure_ascii=False 使 JSON 文件更易读且支持中文
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"生成过程完成。所有描述已保存到 {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 LLaVA 模型为 COCO 验证集图片生成描述。")

    # --- 模型路径参数 ---
    # 根据您的目录结构和需求，这里将 model-path 直接设置为 LLaVA-v1.5-7b 的完整模型路径
    # 如果您要用 LoRA 微调后的模型，请将此路径改为 LoRA 检查点路径，并设置 --model-base
    parser.add_argument("--model-path", type=str,
                        default='/lab/zhangjg_lab/30028000/llava/models/llava-v1.5-7b-fine_tuned_merged',
                        help="LLaVA 模型路径。如果是完整的LLaVA模型，直接指向其目录；如果是LoRA检查点，则指向LoRA目录。")
    parser.add_argument("--model-base", type=str, default=None,
                        help="基础模型路径（例如原始的 LLaVA-v1.5-7b）。当 --model-path 是 LoRA 检查点时需要。")

    # --- 数据路径参数 ---
    parser.add_argument("--coco-val-annotations-file", type=str,
                        default="/lab/zhangjg_lab/30028000/llava/data/coco/annotations/captions_val2014.json",
                        help="COCO 验证集标注文件路径。用于获取图片列表。")
    parser.add_argument("--image-folder", type=str,
                        default="/lab/zhangjg_lab/30028000/llava/data/coco/val2014",
                        help="COCO 验证集图片文件存放的目录。")

    # --- 输出文件参数 ---
    # 动态生成带时间戳的输出文件名，确保不覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式化时间戳 YYYYMMDD_HHMMSS
    default_output_filename = f"./coco_val_results_{timestamp}.json"
    parser.add_argument("--output-file", type=str, default=default_output_filename,
                        help="保存生成结果的输出文件路径。默认值为带有时间戳的文件名。")

    # --- 运行配置参数 ---
    parser.add_argument("--device", type=str, default="cuda",
                        help="指定运行设备，例如 'cuda' 或 'cpu'。")
    parser.add_argument("--conv-mode", type=str, default=None,
                        help="指定对话模式（例如 llava_v1, llava_llama_2, mpt）。如果为 None，将自动推断。")
    parser.add_argument("--load-8bit", action="store_true",
                        help="如果设置，则以 8 位量化加载模型，以节省显存。")
    parser.add_argument("--load-4bit", action="store_true",
                        help="如果设置，则以 4 位量化加载模型，以节省显存。")
    parser.add_argument("--num-test-images", type=int, default=10,
                        help="要测试的图片数量。默认取 COCO 验证集的前 10 张。")

    args = parser.parse_args()

    # 扩展用户路径，处理类似 "~/" 的表示
    args.coco_val_annotations_file = os.path.expanduser(args.coco_val_annotations_file)
    args.image_folder = os.path.expanduser(args.image_folder)
    args.output_file = os.path.expanduser(args.output_file)

    # 打印一些关键参数，方便调试
    print("\n--- 脚本参数概览 ---")
    print(f"模型路径 (--model-path): {args.model_path}")
    print(f"基础模型路径 (--model-base): {args.model_base}")
    print(f"COCO 标注文件: {args.coco_val_annotations_file}")
    print(f"图像文件夹: {args.image_folder}")
    print(f"输出文件: {args.output_file}")
    print(f"设备: {args.device}")
    print(f"加载 8 位: {args.load_8bit}")
    print(f"加载 4 位: {args.load_4bit}")
    print(f"测试图片数量: {args.num_test_images}")
    print("-------------------\n")

    main(args)
