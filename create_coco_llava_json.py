import json
import os

# --- 配置你的路径 ---
# COCO原始标注文件路径
coco_annotation_file = '/lab/zhangjg_lab/30028000/llava/data/coco/annotations/captions_train2014.json'
# COCO图片文件夹路径
coco_image_dir = '/lab/zhangjg_lab/30028000/llava/data/coco/train2014'
# 转换后给LLaVA用的JSON文件的保存路径
output_file = '/lab/zhangjg_lab/30028000/llava/data/coco/annotations/llava_coco_captions_train.json'
# --- 配置结束 ---

# 扩展~符号为用户主目录
coco_annotation_file = os.path.expanduser(coco_annotation_file)
coco_image_dir = os.path.expanduser(coco_image_dir)
output_file = os.path.expanduser(output_file)

print(f"正在读取COCO标注文件: {coco_annotation_file}")
with open(coco_annotation_file, 'r') as f:
    coco_data = json.load(f)

image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
annotations = coco_data['annotations']

llava_data = []
processed_images = set()

# COCO每张图片有5个标注，为了简单快速，我们这里只为每张图片选择一个标注。
# 如果想充分利用数据，可以为每张图片创建5个条目。
for ann in annotations:
    image_id = ann['image_id']

    if image_id in processed_images:
        continue  # 跳过已经处理过的图片

    filename = image_id_to_filename.get(image_id)
    if not filename:
        continue

    # 确保图片文件存在
    image_path = os.path.join(coco_image_dir, filename)
    if not os.path.exists(image_path):
        continue

    caption = ann['caption']

    # 构建LLaVA格式的对话
    # 使用不同的prompt可以引导模型学习不同的任务，这里我们用一个简单的指令
    llava_entry = {
        "id": f"coco_{image_id}",
        "image": os.path.join(os.path.basename(os.path.dirname(image_path)), filename),
        # 使用相对路径 'train2014/filename.jpg'
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nPlease generate a caption for this image."  # <image>是特殊占位符
            },
            {
                "from": "gpt",
                "value": caption
            }
        ]
    }
    llava_data.append(llava_entry)
    processed_images.add(image_id)

# 确保输出目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"转换完成，共处理 {len(llava_data)} 张图片。")
print(f"正在保存到: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(llava_data, f, indent=2, ensure_ascii=False)

print("完成！")
