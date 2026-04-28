# main.py
import os

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# 导入自定义模块
from config import config
from data_preprocessing import preprocess_casia_b
from dataset import GaitGEIDataset
from model import GaitNet
from train_model import train_model
from evaluate_model import evaluate_model


def main():
    """
    主函数：执行完整的步态识别流程
    流程：数据预处理 -> 数据加载 -> 模型创建 -> 训练 -> 评估
    """
    # ===== 步骤0：从视频生成GEI（新增）=====
    if not os.path.exists(config.GEI_OUTPUT_PATH) or len(os.listdir(config.GEI_OUTPUT_PATH)) == 0:
            print("从视频生成GEI...")
            preprocess_from_videos(
                video_dir=config.VIDEO_INPUT_PATH,
                gei_output_dir=config.GEI_OUTPUT_PATH,
                feature_type=config.VIDEO_FEATURE_TYPE,
                image_size=config.GEI_IMAGE_SIZE,
                sequence_length=config.SEQUENCE_LENGTH
            )

    # ==================== 1. 数据预处理阶段 ====================
    print("=" * 50)
    print("步骤1：数据预处理 - 生成步态能量图(GEI)")
    print("=" * 50)
    #首次运行后注释掉，避免重复预处理
    preprocess_casia_b(
        config.RAW_CASIA_A_PATH,  # 参数：原始CASIA-B数据集路径
        config.GEI_OUTPUT_PATH,  # 参数：GEI输出保存路径
        config.GEI_IMAGE_SIZE  # 参数：输出GEI图像尺寸，如(128, 128)
    )
    # ==================== 2. 数据转换定义 ====================
    print("\n" + "=" * 50)
    print("步骤2：定义数据转换")
    print("=" * 50)

    # 定义图像预处理pipeline
    transform = transforms.Compose([
        transforms.Resize(config.GEI_IMAGE_SIZE),  # 统一尺寸，输入：PIL Image，输出：PIL Image
        transforms.ToTensor(),
            # 1.(H, W, C) → (C, H, W)
            # 2.像素值0~255 → 0.0~1.0
            # 3.uint8 → float32
        transforms.Normalize((0.5,), (0.5,))
        # 归一化到[-1,1]，公式：output = (input - 0.5) / 0.5
        # Normalize参数(0.5,0.5)适用于输入范围为[0,1]的情况
    ])
    print(f"图像尺寸: {config.GEI_IMAGE_SIZE}")
    print("数据转换: Resize -> ToTensor -> Normalize")
    # 如果GEI像素值范围不同，需要调整这些参数

    # ==================== 3. 数据集加载与划分 ====================
    print("\n" + "=" * 50)
    print("步骤3：加载数据集")
    print("=" * 50)

    # 创建完整数据集实例
    full_dataset = GaitGEIDataset(
        config.GEI_OUTPUT_PATH,  # 参数：GEI图像存储路径
        transform=transform  # 参数：数据增强和转换操作
    )

    if len(full_dataset) == 0:
        print("错误：数据集为空，无法继续训练。请检查GEI生成过程。")
        return  # 提前退出，避免空数据集导致崩溃

    # 获取类别数量（即行人ID数量）
    num_classes = len(full_dataset.person_to_idx)
    print(f"检测到 {num_classes} 个不同的行人（类别数）。")
    # 痛点说明：某些行人可能图像太少，需要设置最小样本数过滤

    # 划分训练集、验证集、测试集（6:2:2比例）
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    #训练集：越大学习越充分，太小欠拟合
    #验证集：太小结果不稳定，太大训练数据少
    #测试集：太小评估不可靠，太大训练数据少

    # random_split参数说明：
    # - full_dataset: 要划分的数据集
    # - [train_size, val_size, test_size]: 每个子集的大小
    # - generator: 固定随机种子确保结果可复现
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    # 【痛点3】random_split按索引划分，可能导致类别分布不均
    # 建议：使用StratifiedShuffleSplit进行分层采样

    print(f"数据集划分：训练集 {train_size} 张，验证集 {val_size} 张，测试集 {test_size} 张")

    # 创建DataLoader，参数说明：
    # - dataset: 数据集实例
    # - batch_size: 每批数据量，影响内存占用和梯度更新稳定性
    # - shuffle: 是否打乱数据，训练集True，验证/测试集False
    # - num_workers: 并行加载数据的进程数，Windows下常设为0避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"数据加载器创建完成：批次大小 = {config.BATCH_SIZE}")

    # ==================== 4. 模型与优化器创建 ====================
    print("\n" + "=" * 50)
    print("步骤4：创建模型、损失函数和优化器")
    print("=" * 50)

    # 实例化步态识别网络
    # 参数num_classes：输出类别数，等于行人数量
    model = GaitNet(num_classes=num_classes).to(config.DEVICE)

    # 交叉熵损失：适用于多分类问题
    # 输入：模型输出的logits（未经过softmax），输出：损失值
    criterion = nn.CrossEntropyLoss()

    # Adam优化器：自适应学习率优化算法
    # 参数：model.parameters() - 要优化的参数，lr - 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"模型已创建，类别数: {num_classes}")
    print(f"损失函数: 交叉熵损失")
    print(f"优化器: Adam，学习率: {config.LEARNING_RATE}")
    print(f"计算设备: {config.DEVICE}")

    # 创建TensorBoard记录器
    writer = SummaryWriter('runs/gait_experiment')
    # 记录超参数到TensorBoard
    writer.add_hparams(
        {'lr': config.LEARNING_RATE, 'batch_size': config.BATCH_SIZE, 'num_epochs': config.NUM_EPOCHS},
        {'hparam_val': 0}  # 占位值，实际值会在训练时更新
    )
    print(f"TensorBoard 日志保存到: runs/gait_experiment")

    # ==================== 5. 模型训练 ====================
    print("\n" + "=" * 50)
    print(f"步骤5：开始训练（共 {config.NUM_EPOCHS} 个轮次）")
    print("=" * 50)

    # train_model函数参数说明：
    # - model: 待训练的模型
    # - train_loader: 训练数据加载器
    # - criterion: 损失函数
    # - optimizer: 优化器
    # - num_epochs: 训练轮数
    # - device: 计算设备(cpu/cuda)
    train_model(model, train_loader, criterion, optimizer, config.NUM_EPOCHS, config.DEVICE)
    # 💪训练过程缺少验证集监控，可能导致过拟合
    # 建议：在train_model内部增加验证集评估和早停机制

    # ==================== 6. 模型评估 ====================
    print("\n" + "=" * 50)
    print("步骤6：最终评估")
    print("=" * 50)

    # 加载训练过程中保存的最佳模型
    # load_state_dict参数：保存的模型参数字典
    model.load_state_dict(torch.load('best_train_model.pth'))
    print("已加载训练集最佳模型")
    # 【痛点6】模型保存路径硬编码，可能因工作目录变化导致文件未找到
    # 建议：使用config中定义的路径，并添加文件存在性检查

    # 评估验证集
    # evaluate_model函数参数 model: 待评估的模型 val_loader: 验证数据加载器 config.DEVICE: 计算设备
    # # - "验证集": 数据集名称标识，仅用于打印
    # val_accuracy = evaluate_model(model, val_loader, config.DEVICE, "验证集")
    #
    # # 评估测试集
    # test_accuracy = evaluate_model(model, test_loader, config.DEVICE, "测试集")

    # 评估验证集
    val_accuracy, val_details = evaluate_model(model, val_loader, config.DEVICE, "验证集",
                                               return_details=True)

    # 评估测试集
    test_accuracy, test_details = evaluate_model(model, test_loader, config.DEVICE, "测试集",
                                                 return_details=True)

    print(f"CASIA-B数据集上的最终测试准确率: {test_accuracy:.4f}")
    #💪 ** 验证集 **：模型训练时用，用于调超参数、选模型、早停， ** 不参与梯度更新 **
    # ** 测试集 **：模型训练完后用，用于最终评估泛化能力， ** 整个训练过程完全没见过 **
    # ** 关键区别 **：验证集可以反复用（调整模型），测试集 ** 只能用一次 **（出最终结果）。如果根据测试集调模型，测试集就失效了。
    # 记录最终评估结果到TensorBoard
    writer.add_scalar('Final/Test_Accuracy', test_accuracy)
    writer.add_scalar('Final/Val_Accuracy', val_accuracy)
    writer.close()  # 关闭writer，释放资源
    print("TensorBoard 已关闭，运行 tensorboard --logdir=runs 查看")


if __name__ == "__main__":
    """
    脚本入口点
    当直接运行此文件时（而非作为模块导入），执行main函数
    """
    main()