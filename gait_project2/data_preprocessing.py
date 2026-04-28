import cv2
import numpy as np
import os
import glob

#从轮廓图中生成GEI

def generate_gei_from_silhouettes(silhouette_paths, output_gei_path, img_size):
    """从一系列轮廓图片生成步态能量图 (GEI)"""
    silhouettes = []

    for img_path in silhouette_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)

        # 修改1：先归一化到[0,1]范围
        img = img.astype(np.float32) / 255.0

        # 修改2：可选：简单的数据清洗，去除异常帧
        # 如果某帧太暗（轮廓面积太小），跳过
        if np.mean(img) < 0.05:  # 轮廓面积小于5%
            continue

        silhouettes.append(img)

    if len(silhouettes) < 5:  # 修改3：至少需要5帧才生成GEI
        print(f"警告：{output_gei_path} 只有 {len(silhouettes)} 帧，跳过")
        return None

    stacked_silhouettes = np.stack(silhouettes, axis=0)
    gei = np.mean(stacked_silhouettes, axis=0)  # 此时范围在[0,1]

    # 修改4：可选：对比度增强
    # 拉伸到[0,1]完整范围
    gei = (gei - gei.min()) / (gei.max() - gei.min() + 1e-8)

    # 保存时转换回0-255
    gei_uint8 = (gei * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_gei_path), exist_ok=True)
    cv2.imwrite(output_gei_path, gei_uint8)

    return gei

def preprocess_casia_b(raw_data_path, gei_output_path, img_size, force_regenerate=False):
    """
    适配 CASIA-B 数据结构：person/state/angle/*.png
    自动跳过已存在的GEI，避免重复生成

    Args:
        raw_data_path: 原始轮廓数据路径
        gei_output_path: GEI保存路径
        img_size: GEI图像尺寸
        force_regenerate: 是否强制重新生成（默认False，跳过已存在的）
    """
    print(f"开始生成GEI：从 {raw_data_path} 到 {gei_output_path}...")

    # 检查GEI目录是否存在，不存在则创建
    os.makedirs(gei_output_path, exist_ok=True)

    if not os.path.exists(raw_data_path):
        print(f"错误：原始数据路径不存在: {raw_data_path}")
        return

    # 获取所有行人文件夹
    person_ids = [p for p in sorted(os.listdir(raw_data_path))
                  if os.path.isdir(os.path.join(raw_data_path, p))]

    print(f"找到 {len(person_ids)} 个行人: {person_ids[:5]}...")

    gei_count = 0
    skipped_existing = 0
    skipped_empty = 0

    for person_id in person_ids:
        person_path = os.path.join(raw_data_path, person_id)

        # 创建行人GEI子目录
        output_person_dir = os.path.join(gei_output_path, person_id)
        os.makedirs(output_person_dir, exist_ok=True)

        # 遍历行走状态 (nm-01, nm-02, bg-01, cl-01)
        states = [s for s in sorted(os.listdir(person_path))
                  if os.path.isdir(os.path.join(person_path, s))]

        for state in states:
            state_path = os.path.join(person_path, state)

            # 遍历视角 (000, 018, 036...)
            angles = [a for a in sorted(os.listdir(state_path))
                      if os.path.isdir(os.path.join(state_path, a))]

            for angle in angles:
                angle_path = os.path.join(state_path, angle)

                # 构建GEI文件路径
                gei_filename = f"{state}_{angle}.png"
                gei_filepath = os.path.join(output_person_dir, gei_filename)

                # ========== 关键修改：检查GEI是否已存在 ==========
                if not force_regenerate and os.path.exists(gei_filepath):
                    skipped_existing += 1
                    continue  # 跳过已存在的GEI
                # ================================================

                # 获取当前视角下的所有PNG轮廓图像
                silhouette_paths = sorted(glob.glob(os.path.join(angle_path, '*.png')))

                # 跳过空文件夹
                if not silhouette_paths:
                    skipped_empty += 1
                    continue

                # 生成并保存GEI
                gei = generate_gei_from_silhouettes(silhouette_paths, gei_filepath, img_size)
                if gei is not None:
                    gei_count += 1

                    if gei_count % 500 == 0:
                        print(f"已生成 {gei_count} 个GEI...")

    print(f"\nGEI生成完成！")
    print(f"  - 新生成: {gei_count} 个GEI")
    print(f"  - 已存在跳过: {skipped_existing} 个GEI")
    print(f"  - 空文件夹跳过: {skipped_empty} 个")

    if gei_count == 0 and skipped_existing == 0:
        print("错误：没有生成任何GEI。请检查你的原始数据路径和数据结构。")

pass