import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import Counter


class GaitGEIDataset(Dataset):


    def __init__(self, gei_root_dir, transform=None):

        self.gei_root_dir = gei_root_dir
        self.transform = transform
        self.image_paths = []  # 存储所有图像的文件路径
        self.labels = []       # 存储所有图像对应的标签（整数）

        # 获取GEI根目录下所有行人文件夹（按名称排序以保证一致性）
        person_folders = [p for p in sorted(os.listdir(gei_root_dir))
                          if os.path.isdir(os.path.join(gei_root_dir, p))]

        # 创建行人ID到整数标签的映射（如: "001" -> 0, "002" -> 1）
        self.person_to_idx = {person_id: i for i, person_id in enumerate(person_folders)}
        self.num_classes = len(self.person_to_idx)  # 数据集中的行人总数

        print(f"行人标签映射: {self.person_to_idx}")

        # 遍历每个行人文件夹
        for person_id in person_folders:
            person_path = os.path.join(gei_root_dir, person_id)
            if not os.path.isdir(person_path):
                continue

            label = self.person_to_idx[person_id]  # 获取该行人的整数标签

            # 获取该行人文件夹下所有PNG格式的GEI图像
            gei_files = [f for f in os.listdir(person_path) if f.endswith('.png')]

            # 记录每张图像的路径和标签
            for gei_filename in gei_files:
                self.image_paths.append(os.path.join(person_path, gei_filename))
                self.labels.append(label)

        print(f"从 {gei_root_dir} 加载了 {len(self.image_paths)} 张GEI图像，共 {len(person_folders)} 个不同行人。")

        # 统计每个类别的样本数量，用于检查类别平衡性
        label_counts = Counter(self.labels)
        print(f"类别分布: {dict(sorted(label_counts.items()))}")

        # 警告：如果没有找到任何图像
        if len(self.image_paths) == 0:
            print(f"警告: 在 {gei_root_dir} 中找不到GEI图像，请确保GEI生成成功。")

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本

        参数:
            idx: 样本索引

        返回:
            image: 预处理后的图像张量
            label: 对应的行人标签（整数）
        """
        img_path = self.image_paths[idx]  # 获取图像路径
        label = self.labels[idx]          # 获取标签

        # 打开图像并转换为灰度图（GEI本身就是单通道）
        image = Image.open(img_path).convert('L')

        # 应用预处理变换（如转换为张量、归一化等）
        if self.transform:
            image = self.transform(image)

        return image, label