import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class SurtoolDataset(Dataset):
    def __init__(self, txt_file, csv_file, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # 读取train.txt文件中的图片路径
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                self.image_paths.append(line.strip())

        # 读取labels.csv文件中的标签
        df = pd.read_csv(csv_file)
        for path in self.image_paths:
            filename = os.path.basename(path)
            clip_name = filename.split("-")[0]
            label_str = df.loc[df['clip_name'] == clip_name, 'tools_present'].values[0]
            label_list = label_str[1 : -1].split(', ')
            label = [self.map_label(l) for l in label_list if l != "nan"]
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        img = cv2.imread(self.image_paths[idx])
        # print(img)
        #img = cv2.resize(img, (512, 512))
        img = img / 255.0
        img = img.transpose(2, 0, 1)

        # 数据增强
        if self.transform:
            img = self.transform(img)

        return{
            'imgs': img,
            'labels': np.array(self.labels[idx])
        }

    def map_label(self, label_name):
        # 将类别名称映射为数字标签
        label_map = {
            'needle driver': 0,
            'cadiere forceps': 1,
            'prograsp forceps': 2,
            'grasping retractor': 3,
            'clip applier ': 4,
            'bipolar dissector': 5,
            'bipolar forceps': 6,
            'stapler': 7,
            'monopolar curved scissors': 8,
            'force bipolar': 9,
            'permanent cautery hook/spatula': 10,
            'suction irrigator': 11,
            'vessel sealer': 12,
            'tip-up fenestrated grasper': 13
        }
        return label_map[label_name]

