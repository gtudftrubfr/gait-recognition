import torch
from config import config
from torch.utils.tensorboard import SummaryWriter
#训练模型，并保存训练集上最好的模型

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型，并保存训练集上最好的模型

    参数说明：
    ----------
    model : torch.nn.Module
        待训练的神经网络模型
    train_loader : torch.utils.data.DataLoader
        训练数据加载器，提供批次化的训练数据
    criterion : torch.nn.modules.loss._Loss
        损失函数，用于计算预测值与真实值的差异
    optimizer : torch.optim.Optimizer
        优化器，用于更新模型参数
    num_epochs : int
        训练的轮数（遍历整个数据集的次数）
    device : str or torch.device
        计算设备，如 'cuda' 或 'cpu'

    返回值：
    --------
    None（无返回值，但会保存最佳模型到文件）

    痛点说明：
    ----------
    1. 缺少验证集监控，容易过拟合
    2. 没有学习率调度器
    3. TensorBoard重复创建
    4. 缺少梯度裁剪防止梯度爆炸
    5. 缺少训练中断恢复机制
    6. 模型保存路径硬编码
    """

    # ==================== 模型训练模式设置 ====================
    # 设置为训练模式：启用Dropout、BatchNorm等训练专用层
    model.train()
    print("开始训练了...")

    # 【痛点1】TensorBoard重复创建问题
    # 问题：如果在main.py中已经创建了writer，这里又创建新的，会导致日志重复或冲突
    # 解决方案：将writer作为参数传入，或者使用全局单例
    writer = SummaryWriter('runs/gait_experiment')

    # 初始化最佳训练准确率跟踪变量
    best_train_acc = 0.0

    # 【痛点2】global_step变量未使用
    # 问题：定义了global_step但没有使用，可能是预留的步数计数器
    # 建议：用于记录总训练步数，在批次级别记录指标时使用
    global_step = 0

    # ==================== 训练主循环 ====================
    for epoch in range(num_epochs):
        # 每个epoch的训练指标初始化
        running_loss = 0.0  # 累计损失值（未平均）
        correct_predictions = 0  # 正确预测的总数
        total_samples = 0  # 已处理的样本总数

        # ==================== 批次训练循环 ====================
        for i, (inputs, labels) in enumerate(train_loader):
            # 将数据移动到指定设备（GPU或CPU）
            inputs, labels = inputs.to(device), labels.to(device)

            # 【痛点4】梯度累积风险
            # 说明：optimizer.zero_grad()清空之前的梯度
            # 如果忘记调用，梯度会累积导致训练失败
            optimizer.zero_grad()

            # 前向传播：计算模型预测
            outputs = model(inputs)

            # 计算损失值
            loss = criterion(outputs, labels)

            # 反向传播：计算梯度
            loss.backward()

            # 【痛点5】缺少梯度裁剪
            # 问题：梯度爆炸可能导致训练不稳定
            # 解决方案：添加 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新模型参数
            optimizer.step()

            # 累积损失值（乘以batch大小用于后续加权平均）
            running_loss += loss.item() * inputs.size(0)

            # 获取预测类别（最大值对应的索引）
            _, predicted = torch.max(outputs.data, 1)

            # 更新统计信息
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels.to(predicted.device)).sum().item()

        # ==================== Epoch结束统计 ====================
        # 计算当前epoch的平均损失和准确率
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples

        # 打印训练进度
        print(f"轮次{epoch + 1}/{num_epochs}, 损失值: {epoch_loss:.4f}, 正确率: {epoch_acc:.4f}")

        # 记录到TensorBoard
        writer.add_scalar('Epoch/Train_Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Train_Accuracy', epoch_acc, epoch)

        # 【痛点6】缺少学习率调度
        # 问题：固定学习率可能导致收敛缓慢或震荡
        # 建议：添加 scheduler.step() 如 ReduceLROnPlateau 或 StepLR

        # ==================== 模型保存策略 ====================
        # 保存训练集上准确率最高的模型
        if epoch_acc > best_train_acc:
            best_train_acc = epoch_acc
            # 【痛点7】硬编码保存路径
            # 问题：模型保存路径固定为当前目录，容易丢失或覆盖
            # 建议：使用 config.MODEL_SAVE_PATH，并添加时间戳
            torch.save(model.state_dict(), 'test_best.pth')
            print(f"保存训练集最佳模型，准确率: {epoch_acc:.4f}")

    # ==================== 训练结束 ====================
    print(f"训练完成！最佳训练准确率: {best_train_acc:.4f}")
    writer.close()

    # 【痛点9】缺少模型保存最佳实践
    # 问题：只保存了state_dict，没有保存优化器状态、epoch等信息
    # 建议：保存完整的checkpoint用于恢复训练
    # checkpoint = {
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'best_acc': best_train_acc,
    # }
    # torch.save(checkpoint, 'checkpoint.pth')