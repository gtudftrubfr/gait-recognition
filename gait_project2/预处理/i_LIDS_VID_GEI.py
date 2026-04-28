import cv2
import numpy as np
import os
from tqdm import tqdm

# ===================== 【仅修改这2个路径】 =====================
# 你生成的319个视频路径（和视频代码的OUTPUT_DIR一致）
VIDEO_DIR = r"D:\gaitset\i-LIDS-VID_generated_videos"
# GEI图片保存路径
GEI_OUTPUT_DIR = r"D:\gaitset\i-LIDS-VID-GEI"
# ==============================================================

# GEI参数（和你原有代码一致）
IMAGE_SIZE = (64, 64)  # GEI尺寸
MIN_FRAMES = 5  # 最少5帧生成GEI


def extract_silhouette_from_frame(frame):
    """从视频帧提取人体轮廓（适配步态GEI预处理）"""
    # 转灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # 二值化（i-LIDS-VID是剪影图，直接阈值处理）
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def video_to_silhouette_list(video_path):
    """读取视频，提取所有帧的轮廓图"""
    cap = cv2.VideoCapture(video_path)
    silhouettes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 提取轮廓
        sil = extract_silhouette_from_frame(frame)
        silhouettes.append(sil)

    cap.release()
    return silhouettes


# ===================== 【完全保留你原始的GEI生成算法】 =====================
def generate_gei_from_silhouettes(silhouettes, output_gei_path, img_size):
    """从轮廓帧生成GEI（你的原版核心逻辑，无修改）"""
    processed_sils = []

    for img in silhouettes:
        # 缩放尺寸
        img = cv2.resize(img, img_size)
        # 归一化到 [0,1]
        img = img.astype(np.float32) / 255.0
        # 过滤无效帧（轮廓面积太小）
        if np.mean(img) < 0.05:
            continue
        processed_sils.append(img)

    # 帧数不足跳过
    if len(processed_sils) < MIN_FRAMES:
        print(f"警告：{os.path.basename(output_gei_path)} 有效帧不足，跳过")
        return None

    # 计算GEI
    stacked = np.stack(processed_sils, axis=0)
    gei = np.mean(stacked, axis=0)
    # 对比度增强
    gei = (gei - gei.min()) / (gei.max() - gei.min() + 1e-8)
    # 转8位图像
    gei_uint8 = (gei * 255).astype(np.uint8)

    # 保存
    os.makedirs(os.path.dirname(output_gei_path), exist_ok=True)
    cv2.imwrite(output_gei_path, gei_uint8)
    return gei


# ===================== 批量处理319个视频 =====================
def batch_generate_gei_from_videos():
    """批量处理所有视频，一对一生成GEI"""
    os.makedirs(GEI_OUTPUT_DIR, exist_ok=True)

    # 获取所有视频文件（001.mp4 ~ 319.mp4）
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    total = len(video_files)
    success = 0
    skip = 0

    print(f"开始处理：共 {total} 个视频 → 生成 {total} 个GEI")
    print("=" * 60)

    for video_name in tqdm(video_files, desc="生成GEI"):
        # 视频编号：001.mp4 → 001
        video_id = os.path.splitext(video_name)[0]
        # 视频完整路径
        video_path = os.path.join(VIDEO_DIR, video_name)
        # GEI保存路径
        gei_path = os.path.join(GEI_OUTPUT_DIR, f"{video_id}.png")

        # 提取视频所有轮廓帧
        sils = video_to_silhouette_list(video_path)
        if not sils:
            skip += 1
            continue

        # 生成GEI
        gei = generate_gei_from_silhouettes(sils, gei_path, IMAGE_SIZE)
        if gei is not None:
            success += 1

    print("=" * 60)
    print(f"处理完成！")
    print(f"✅ 成功生成GEI：{success} 个")
    print(f"⚠️  跳过无效视频：{skip} 个")
    print(f"🎯 总视频数：{total} 个")
    print(f"GEI保存路径：{GEI_OUTPUT_DIR}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    batch_generate_gei_from_videos()