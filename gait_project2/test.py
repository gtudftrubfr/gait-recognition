import os
#临时测试
import pandas as pd

if os.path.exists('confusion_matrix_测试集.csv'):
    cm = pd.read_csv('confusion_matrix_测试集.csv', index_col=0)
    print("混淆矩阵形状:", cm.shape)
    print("\n前10行前10列:")
    print(cm.iloc[:10, :10])
else:
    print("未找到混淆矩阵文件，请先运行保存代码")