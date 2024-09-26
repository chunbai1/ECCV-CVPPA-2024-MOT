import os

# 定义输入和输出目录
input_dir = '/data/ChaiJM/Competition/CVPPA-DMOT/Code/ByteTrack-main/results/4ep'  # 替换为你的输入目录路径
output_dir = input_dir + '_fuzhi'  # 替换为你的输出目录路径

# 如果输出目录不存在，创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                values = line.strip().split(',')
                img_id, obj_id = values[0], values[1]
                x, y = max(float(values[2]), 0), max(float(values[3]), 0)
                width, height, confidence = values[4], values[5], values[6]

                # 将修改后的值写入输出文件
                outfile.write(f"{img_id},{obj_id},{x:.2f},{y:.2f},{width},{height},{confidence}\n")
