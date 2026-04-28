# config.py
#路径，数据配置
import torch
import random
import numpy as np


class Config:
    # --- 文件路径配置 ---
    RAW_CASIA_A_PATH = r'D:\gaitset\GaitDatasetB-silh'
    GEI_OUTPUT_PATH = r'D:\gaitset\GaitDatasetB-GEI'
    VIDEO_FEATURE_TYPE = 'silhouette'
    SEQUENCE_LENGTH = 40

    # --- 数据处理配置 ---
    GEI_IMAGE_SIZE = (64, 64)
    TRAIN_RATIO = 0.9 #


    # --- 训练配置 ---
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 5
    #WEIGHT_DECAY = 0.001  # 新增
    #EARLY_STOPPING_PATIENCE = 10  # 新增

    # --- 系统配置 ---
    RANDOM_SEED = 7
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 初始化随机种子，确保实验可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 创建配置实例并设置种子
config = Config()
set_seed(config.RANDOM_SEED)

print(f"Using device: {config.DEVICE}")