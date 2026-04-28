import torch
import numpy as np
#评估模型
def evaluate_model(model, data_loader, device, data_name="验证集",return_details=False):
    """
    评估模型在数据集上的性能

    参数说明：
    ----------
    model : torch.nn.Module
        待评估的神经网络模型，必须已经加载预训练权重
    data_loader : torch.utils.data.DataLoader
        数据加载器，提供批次化的评估数据
    device : str or torch.device
        计算设备，如 'cuda'、'cuda:0' 或 'cpu'
    data_name : str, optional
        数据集名称标识，仅用于打印输出，默认值为"验证集"

    返回值：
    --------
    accuracy : float
        模型在数据集上的分类准确率，范围 [0.0, 1.0]

  
    """

    # ==================== 模型评估模式设置 ====================
    # 设置为评估模式：禁用Dropout、使用BatchNorm的累积统计量
    model.eval()

    correct_predictions = 0  # 正确预测的样本数
    total_samples = 0  # 总样本数
    all_predictions = []  # 存储所有预测
    all_labels = []  # 存储所有真实标签
    top3_correct = 0  # Top-3正确数
    top5_correct = 0  # Top-5正确数

    if len(data_loader) == 0: return 0.0

    print(f"开始评估 on {data_name}...")

    with torch.no_grad():
        # ==================== 批次循环 ====================
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            # 将数据移动到指定设备（GPU或CPU）
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.max(outputs.data, 1)


            # 添加Top-3/Top-5计算
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            for i in range(labels.size(0)):
                if labels[i] in top5_pred[i]:
                    top5_correct += 1
                    if labels[i] in top5_pred[i][:3]:
                        top3_correct += 1


            # ：正确接收两个返回值
            values, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()


            #对大数据集添加进度条，如使用tqdm库
            if batch_idx % 100 == 0:
                print(f"Processed {total_samples} samples...")

    if total_samples == 0:
        print(f"警告：{data_name}为空，无法计算准确率")
        return 0.0

    accuracy = correct_predictions / total_samples
    print(f"{data_name}准确率: {accuracy:.4f}")

 

    #添加计算召回率、精确率、F1、混淆矩阵
    if return_details:
        from collections import defaultdict
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        num_classes = len(np.unique(all_labels))


        # 初始化混淆矩阵
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for true, pred in zip(all_labels, all_predictions):
            confusion_matrix[true, pred] += 1

        # 计算每个类别的指标
        precision = {}
        recall = {}
        f1_score = {}
        for c in range(num_classes):
            tp = confusion_matrix[c, c]
            fp = confusion_matrix[:, c].sum() - tp
            fn = confusion_matrix[c, :].sum() - tp

            precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[
                c]) > 0 else 0.0

        # 计算Top-K准确率
        top1_acc = correct_predictions / total_samples
        top3_acc = top3_correct / total_samples
        top5_acc = top5_correct / total_samples

        # 打印结果
        print(f"\nTop-1准确率: {top1_acc:.4f}")
        print(f"Top-3准确率: {top3_acc:.4f}")
        print(f"Top-5准确率: {top5_acc:.4f}")
        print(f"宏平均精确率: {np.mean(list(precision.values())):.4f}")
        print(f"宏平均召回率: {np.mean(list(recall.values())):.4f}")
        print(f"宏平均F1分数: {np.mean(list(f1_score.values())):.4f}")

        # 保存混淆矩阵到CSV文件
        import pandas as pd
        cm_df = pd.DataFrame(confusion_matrix)
        cm_df.to_csv(f'confusion_matrix_{data_name}.csv')
        print(f" 混淆矩阵已保存到: confusion_matrix_{data_name}.csv")
        print(f"混淆矩阵形状: {confusion_matrix.shape}")

        # 打印前10行预览
        print(f"\n前10行前10列预览:")
        print(cm_df.iloc[:10, :10].to_string())

        # 返回详细指标
        return accuracy, {
            'top1_accuracy': top1_acc,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': confusion_matrix
        }

    return accuracy
       # 少其他评估指标
    # 添加以下指标的返回选项
    # - 召回率 (Recall)
    # - 精确率 (Precision)
    # - F1分数 (F1-Score)
    # - 混淆矩阵 (Confusion Matrix)
    # - Top-3/Top-5准确率
