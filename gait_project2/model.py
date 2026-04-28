# model.py
#模型参数配置
import torch
import torch.nn as nn


class GaitNet(nn.Module):
    """
    一个简单的卷积神经网络 (CNN) 用于步态识别。
    输入为单通道GEI图像，输出为行人ID的分类分数。
    """

    def __init__(self, num_classes):
        super(GaitNet, self).__init__()
        self.features = nn.Sequential(
            # 第1层：64x64 -> 32x32
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 批量归一化，对64个通道归一化，加速收敛稳定训练
            nn.ReLU(inplace=True),#ReLU激活函数，inplace=True节省内存，公式max(0,x)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第2层：32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第3层：16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 新增第4层：8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # 第1层：64x64 -> 32x32
            # nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(0.25),  # 2D Dropout对卷积有效
            #
            # # 第2层：32x32 -> 16x16
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(0.25),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#自适应平均池化，不管输入多大，输出都是1x1的特征图


        # # 分类器也加深
        self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 512),  # 增大隐藏层
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, num_classes)

        # 简化分类器
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(64, 128),  # 更小的隐藏层
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, num_classes)
        # )
    )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平特征图，准备输入全连接层
        x = self.classifier(x)
        return x
