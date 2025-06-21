import argparse
import torch
from datetime import datetime
import json
import os
from tqdm import tqdm
from collections import defaultdict
import math

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import nltk
nltk.data.path.append('/home/moting/llava_project/nltk_data')
# 导入 NLTK 用于 BLEU 计算
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize


# 确保你已经下载了 punkt 分词器
# import nltk
# nltk.download('punkt')

def calculate_metrics(annotation_file, result_file):
    """
    计算给定标注文件和结果文件的 BLEU-1/2/3/4 和简化的 CIDEr 分数。
    """
    print("\n--- 开始评估指标 ---")

    # 加载参考标注
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            coco_annotations = json.load(f)
    except FileNotFoundError:
        print(f"错误: 标注文件未找到于 {annotation_file}")
        return {}
    except json.JSONDecodeError:
        print(f"错误: 无法解析标注文件 {annotation_file}。请检查 JSON 格式。")
        return {}

    # 整理参考标注：{image_id: [caption1, caption2, ...]}
    references = defaultdict(list)
    for ann in coco_annotations.get('annotations', []):  # 使用 .get() 避免键错误
        references[ann['image_id']].append(ann['caption'])

    # 加载模型生成的描述
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            generated_results = json.load(f)
    except FileNotFoundError:
        print(f"错误: 结果文件未找到于 {result_file}")
        return {}
    except json.JSONDecodeError:
        print(f"错误: 无法解析结果文件 {result_file}。请检查 JSON 格式。")
        return {}

    # 整理生成结果：{image_id: caption}
    hypotheses = {item['image_id']: item['caption'] for item in generated_results}

    # 筛选出同时存在于参考和生成结果中的图片ID
    common_image_ids = sorted(list(set(references.keys()) & set(hypotheses.keys())))
    if not common_image_ids:
        print("没有找到共同的图片ID进行评估。请检查文件内容。")
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0, "CIDEr": 0.0}

    # --- BLEU 计算 ---
    print("计算 BLEU-1, BLEU-2, BLEU-3, BLEU-4...")
    bleu_scores_all_n = defaultdict(list)  # 存储每个 n 阶的 BLEU 分数列表
    chencherry = SmoothingFunction()  # 使用平滑函数

    for img_id in tqdm(common_image_ids, desc="计算 BLEU"):
        ref_caps_tokenized = [word_tokenize(cap.lower()) for cap in references[img_id]]
        hyp_cap_tokenized = word_tokenize(hypotheses[img_id].lower())

        # 计算 BLEU-1 到 BLEU-4
        weights_list = [
            (1.0, 0.0, 0.0, 0.0),  # BLEU-1
            (0.5, 0.5, 0.0, 0.0),  # BLEU-2
            (0.33, 0.33, 0.33, 0.0),  # BLEU-3
            (0.25, 0.25, 0.25, 0.25)  # BLEU-4
        ]
        for i, weights in enumerate(weights_list):
            n = i + 1
            score = sentence_bleu(ref_caps_tokenized, hyp_cap_tokenized, weights=weights,
                                  smoothing_function=chencherry.method1)
            bleu_scores_all_n[f"BLEU-{n}"].append(score)

    final_metrics = {}
    for n_gram_type, scores in bleu_scores_all_n.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        final_metrics[n_gram_type] = avg_score
        print(f"{n_gram_type}: {avg_score:.4f}")

    # --- 简化的 CIDEr 计算 ---
    print("计算简化的 CIDEr...")

    cider_scores = []

    # 辅助函数：提取 n-grams (这里只考虑 uni-gram 和 bi-gram 为简化)
    def get_ngrams(tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(' '.join(tokens[i:i + n]))
        return ngrams

    # 辅助函数：计算 TF (Term Frequency)
    def compute_tf(tokens):
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        return tf

    # 辅助函数：计算余弦相似度
    def cosine_similarity(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])

        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        return numerator / denominator

    # 收集所有 n-grams 用于 IDF 计算 (这里只用 uni-gram 作为简化)
    all_ngrams_corpus = defaultdict(int)
    doc_count = 0
    for img_id in common_image_ids:
        doc_count += 1
        unique_ngrams_per_doc = set()
        for cap in references[img_id]:
            unique_ngrams_per_doc.update(get_ngrams(word_tokenize(cap.lower()), 1))
        unique_ngrams_per_doc.update(get_ngrams(word_tokenize(hypotheses[img_id].lower()), 1))

        for ngram in unique_ngrams_per_doc:
            all_ngrams_corpus[ngram] += 1

    # 计算 IDF (Inverse Document Frequency)
    idf_scores = {ngram: math.log(doc_count / (count + 1)) for ngram, count in all_ngrams_corpus.items()}

    for img_id in tqdm(common_image_ids, desc="计算 CIDEr"):
        ref_caps_tokenized = [word_tokenize(cap.lower()) for cap in references[img_id]]
        hyp_cap_tokenized = word_tokenize(hypotheses[img_id].lower())

        current_img_cider_scores = []
        for ref_cap_tokens in ref_caps_tokenized:
            ref_ngrams_1 = get_ngrams(ref_cap_tokens, 1)
            ref_ngrams_2 = get_ngrams(ref_cap_tokens, 2)
            hyp_ngrams_1 = get_ngrams(hyp_cap_tokenized, 1)
            hyp_ngrams_2 = get_ngrams(hyp_cap_tokenized, 2)

            ref_tf_idf_vec = defaultdict(float)
            for ngram in ref_ngrams_1:
                ref_tf_idf_vec[ngram] += (1 * idf_scores.get(ngram, 0))
            for ngram in ref_ngrams_2:
                ref_tf_idf_vec[ngram] += (1 * idf_scores.get(ngram, 0))

            hyp_tf_idf_vec = defaultdict(float)
            for ngram in hyp_ngrams_1:
                hyp_tf_idf_vec[ngram] += (1 * idf_scores.get(ngram, 0))
            for ngram in hyp_ngrams_2:
                hyp_tf_idf_vec[ngram] += (1 * idf_scores.get(ngram, 0))

            sim = cosine_similarity(ref_tf_idf_vec, hyp_tf_idf_vec)
            current_img_cider_scores.append(sim)

        if current_img_cider_scores:
            cider_scores.append(max(current_img_cider_scores))
        else:
            cider_scores.append(0.0)

    avg_cider = sum(cider_scores) / len(cider_scores) if cider_scores else 0.0
    final_metrics["CIDEr"] = avg_cider
    print(f"CIDEr (简化): {avg_cider:.4f}")

    print("--- 评估完成 ---")
    return final_metrics


def main(args):
    # 禁用 PyTorch 初始化时的冗余信息，避免干扰
    disable_torch_init()

    # --- 加载模型 ---
    model_name = get_model_name_from_path(args.model_path)
    print(f"正在从 {args.model_path} 加载模型...")
    print(f"模型名称推断为: {model_name}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base if args.model_base else None,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device
    )
    print("模型加载完成。")

    # --- 确定对话模式 ---
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            f'[警告] 自动确定的对话模式是 {conv_mode}，而命令行指定了 --conv-mode={args.conv_mode}，将使用 {args.conv_mode}。')
        conv_mode = args.conv_mode
    else:
        args.conv_mode = conv_mode

    # --- 准备验证集数据 ---
    print(f"正在加载 COCO 验证集标注文件: {args.coco_val_annotations_file}")
    with open(args.coco_val_annotations_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    val_images = {img['id']: img for img in val_data['images']}
    all_image_ids = list(val_images.keys())

    # test_image_ids = all_image_ids
    test_image_ids = all_image_ids[:args.num_test_images]
    print(f"将测试前 {len(test_image_ids)} 张图片。")

    results = []

    # --- 构建批次并生成描述 ---
    # 新增：定义 batch_size 参数
    batch_size = args.batch_size

    # 将图像ID分批
    batches = [test_image_ids[i:i + batch_size] for i in range(0, len(test_image_ids), batch_size)]

    for batch_idx, batch_image_ids in enumerate(tqdm(batches, desc="生成图片描述 (批处理)")):
        batch_image_tensors = []
        batch_input_ids = []
        current_batch_image_infos = []

        # 收集当前批次的图像信息和处理图像
        for image_id in batch_image_ids:
            image_info = val_images[image_id]
            image_file = os.path.join(args.image_folder, image_info['file_name'])

            if not os.path.exists(image_file):
                print(f"警告：图像文件 {image_file} 不存在，跳过该图片。")
                continue

            # 将有效的图片信息添加到当前批次中
            current_batch_image_infos.append(image_info)

            # 图像预处理
            image = Image.open(image_file).convert('RGB')
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)
            batch_image_tensors.append(image_tensor)

            # 文本提示处理
            qs = "<image> Please generate a description for the image."
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).to(args.device)
            batch_input_ids.append(input_ids)

        # 检查批次是否为空 (例如，所有文件都不存在)
        if not batch_image_tensors:
            continue

        # 将批次内的图像张量和输入ID堆叠起来
        # 注意：这里假设所有图片经过预处理后维度相同。如果不同，需要进行padding或resize操作。
        # 对于LLaVA通常会将其resize到固定尺寸，所以这里直接stack是可行的。
        images_batch = torch.cat(batch_image_tensors, dim=0)  # 沿batch维度拼接
        input_ids_batch = torch.cat(batch_input_ids, dim=0)  # 沿batch维度拼接

        # 停止词处理（针对批处理，需要确保每个样本都有对应的停止逻辑）
        # 对于KeywordsStoppingCriteria，它会处理批次中的所有序列
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[
                                                             args.conv_mode].sep_style != SeparatorStyle.TWO else \
        conv_templates[args.conv_mode].sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids_batch)

        with torch.inference_mode():
            output_ids_batch = model.generate(
                input_ids_batch,
                images=images_batch,
                do_sample=False,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        # 解码批次结果
        for i, output_ids in enumerate(output_ids_batch):
            outputs = tokenizer.decode(output_ids, skip_special_tokens=False)
            outputs = outputs.strip().removeprefix("<s>").removesuffix("</s>").strip()

            # 移除 <unk> 标记，以及任何多余的空格
            outputs = outputs.replace("<unk>", "").strip()
            # 再次清理由于移除 <unk> 可能产生的多余空格
            outputs = ' '.join(outputs.split())
            outputs = outputs.strip().removeprefix("<s>").removesuffix("</s>").strip()

            # 使用当前批次中对应的image_info
            image_id = current_batch_image_infos[i]['id']
            results.append({"image_id": image_id, "caption": outputs})

    # --- 保存结果 ---
    print(f"所有描述已生成，准备保存到 {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"生成过程完成。所有描述已保存到 {args.output_file}")

    # --- 计算评估指标 ---
    metrics = calculate_metrics(args.coco_val_annotations_file, args.output_file)
    print("\n--- 评估结果 ---")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")
    print("-------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 LLaVA 模型为 COCO 验证集图片生成描述。")

    parser.add_argument("--model-path", type=str,
                        default='/home/moting/llava_project/models/llava-v1.5-7b',
                        help="LLaVA 模型路径。如果是完整的LLaVA模型,直接指向其目录,如果是LoRA检查点,则指向LoRA目录。")
    parser.add_argument("--model-base", type=str, default=None,
                        help="基础模型路径（例如原始的 LLaVA-v1.5-7b)。当 --model-path 是 LoRA 检查点时需要。")

    parser.add_argument("--coco-val-annotations-file", type=str,
                        default="/home/moting/llava_project/coco2014/annotations/annotations/captions_val2014.json",
                        help="COCO 验证集标注文件路径。用于获取图片列表。")
    parser.add_argument("--image-folder", type=str,
                        default="/home/moting/llava_project/coco2014/val2014",
                        help="COCO 验证集图片文件存放的目录。")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_filename = f"/home/moting/llava_project/eval_results_original/original_eval_results_{timestamp}.json"
    parser.add_argument("--output-file", type=str, default=default_output_filename,
                        help="保存生成结果的输出文件路径。默认值为带有时间戳的文件名。")

    parser.add_argument("--device", type=str, default="cuda",
                        help="指定运行设备，例如 'cuda' 或 'cpu'。")
    parser.add_argument("--conv-mode", type=str, default=None,
                        help="指定对话模式（例如 llava_v1, llava_llama_2, mpt）。如果为 None，将自动推断。")
    parser.add_argument("--load-8bit", action="store_true",
                        help="如果设置，则以 8 位量化加载模型，以节省显存。")
    parser.add_argument("--load-4bit", action="store_true",
                        help="如果设置，则以 4 位量化加载模型，以节省显存。")
    parser.add_argument("--num-test-images", type=int, default=32000,
                        help="要测试的图片数量。默认取 COCO 验证集的前 x 张。")
    # 新增：批处理大小参数
    
    
    parser.add_argument("--batch-size", type=int, default=16,
                        help="每次推理的图片数量。增加此值可以提高推理速度。")

    args = parser.parse_args()

    args.coco_val_annotations_file = os.path.expanduser(args.coco_val_annotations_file)
    args.image_folder = os.path.expanduser(args.image_folder)
    args.output_file = os.path.expanduser(args.output_file)

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
    print(f"批处理大小 (--batch-size): {args.batch_size}")  # 新增打印
    print("-------------------\n")

    main(args)
