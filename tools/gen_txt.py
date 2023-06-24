import os
import random

# 定义图片目录和输出目录
image_dir = '/data2/zhenglujie/data/SurtoolDataset/images/'
output_dir = '/data2/zhenglujie/data/SurtoolDataset/'

# 获取所有图片的路径
images = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, img) for img in images]
print(len(image_paths))

# 随机打乱图片路径列表
random.shuffle(image_paths)

# 计算训练集和测试集的分割点
split = int(0.8 * len(image_paths)) # 80%用作训练集，20%用作测试集

# 将图片路径列表分割成训练集和测试集
train_paths = image_paths[:split]
test_paths = image_paths[split:]

# 将训练集和测试集的图片路径写入train.txt和test.txt文件中
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_paths))

with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_paths))