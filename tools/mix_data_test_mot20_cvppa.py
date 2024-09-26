import json
import os


"""
cd datasets
mkdir -p mix_mot20_ch/annotations
cp MOT20/annotations/val_half.json mix_mot20_ch/annotations/val_half.json
cp MOT20/annotations/test.json mix_mot20_ch/annotations/test.json
cd mix_mot20_ch
ln -s ../MOT20/train mot20_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
cd ..
"""

# 读取 MOT20 数据集的 JSON 文件
mot_json = json.load(open('datasets/MOT20_CVPPA/annotations/train.json', 'r'))

# 初始化图像和标注的列表
img_list = list()
for img in mot_json['images']:
    img['file_name'] = 'mot20_train/' + img['file_name']
    img_list.append(img)

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

# 保留视频信息和类别信息
video_list = mot_json['videos']
category_list = mot_json['categories']

# 创建最终合并的 JSON 文件
mix_json = {
    'images': img_list,
    'annotations': ann_list,
    'videos': video_list,
    'categories': category_list
}

# 将合并后的数据保存到新的 JSON 文件中
json.dump(mix_json, open('datasets/mix_mot20_cvppa/annotations/train.json', 'w'))

print('MOT20-CVPPA 数据集处理完成')