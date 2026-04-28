import torch
from torchvision import transforms
from PIL import Image, ImageTk
from model import GaitNet
from config import config
from dataset import GaitGEIDataset
import tkinter as tk
from tkinter import filedialog, messagebox
#测试一个图片看是否正确
# ---------------------- 模型部分（原代码） ----------------------
def load_model(model_path, num_classes, device):
    model = GaitNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(config.GEI_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return image_tensor

def predict(image_path, model, person_to_idx, device):
    idx_to_person = {idx: person_id for person_id, idx in person_to_idx.items()}
    image_tensor = preprocess(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    person_id = idx_to_person[predicted_idx]
    return person_id, confidence

# ---------------------- GUI界面部分 ----------------------
class GaitRecognitionGUI:
    def __init__(self, root, model, person_to_idx, device):
        self.root = root
        self.root.title("步态识别系统")
        self.root.geometry("600x500")

        # 模型和映射
        self.model = model
        self.person_to_idx = person_to_idx
        self.device = device
        self.img_path = None  # 保存选择的图片路径

        # 界面布局
        self.create_widgets()

    def create_widgets(self):
        # 1. 标题
        title_label = tk.Label(self.root, text="步态识别系统", font=("黑体", 18, "bold"))
        title_label.pack(pady=10)

        # 2. 文件选择部分
        frame1 = tk.Frame(self.root)
        frame1.pack(pady=5)

        tk.Label(frame1, text="待预测行人的GEI图片路径:").pack(side=tk.LEFT, padx=5)
        self.path_var = tk.StringVar()
        path_entry = tk.Entry(frame1, textvariable=self.path_var, width=40)
        path_entry.pack(side=tk.LEFT, padx=5)

        browse_btn = tk.Button(frame1, text="浏览", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=5)

        # 3. 图片显示区域
        self.img_label = tk.Label(self.root, text="GEI图片将显示在这里", bg="white", width=200, height=200)
        self.img_label.pack(pady=10)

        # 4. 识别按钮
        self.rec_btn = tk.Button(self.root, text="进行识别", command=self.run_recognition, font=("黑体", 14), bg="#4CAF50", fg="white")
        self.rec_btn.pack(pady=10)

        # 5. 结果显示区域
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10)

        self.real_label = tk.Label(result_frame, text="真实标签: --", font=("黑体", 15))
        self.real_label.pack(pady=5)

        self.pred_label = tk.Label(result_frame, text="预测标签: --", font=("黑体", 15))
        self.pred_label.pack(pady=5)

        self.conf_label = tk.Label(result_frame, text="概率: --", font=("黑体", 15))
        self.conf_label.pack(pady=5)

    def browse_file(self):
        # 打开文件选择框，只选图片
        file_path = filedialog.askopenfilename(
            title="选择GEI图片",
            filetypes=[("PNG图片", "*.png"), ("JPG图片", "*.jpg"), ("所有文件", "*.*")]
        )
        if file_path:
            self.img_path = file_path
            self.path_var.set(file_path)
            # 显示图片
            self.show_image(file_path)

    def show_image(self, path):
        # 读取并缩放图片，显示到界面
        img = Image.open(path).convert('L')
        img = img.resize((200, 200), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk  # 防止被垃圾回收

    def run_recognition(self):
        if not self.img_path:
            messagebox.showwarning("警告", "请先选择一张GEI图片！")
            return

        try:
            # 调用预测函数
            person_id, confidence = predict(self.img_path, self.model, self.person_to_idx, self.device)

            # 更新界面结果
            self.pred_label.config(text=f"预测标签: {person_id}", fg="green")
            self.conf_label.config(text=f"概率: {confidence:.5f} ({confidence*100:.2f}%)", fg="blue")

            # 真实标签这里可以手动填，也可以根据文件名解析（可选）
            # 示例：从文件名解析真实标签，比如 001-bg-02-018-002.png → 001
            real_id = self.img_path.split('/')[-1].split('-')[0]
            self.real_label.config(text=f"真实标签: {real_id}", fg="black")

        except Exception as e:
            messagebox.showerror("错误", f"识别失败：{str(e)}")

# ---------------------- 主函数启动 ----------------------
if __name__ == "__main__":
    # 1. 加载数据集映射和模型
    temp_dataset = GaitGEIDataset(config.GEI_OUTPUT_PATH, transform=None)
    person_to_idx = temp_dataset.person_to_idx
    num_classes = len(person_to_idx)

    # 2. 加载训练好的模型
    model_path = "../best_train_model.pth"  # 改成你的模型路径
    model = load_model(model_path, num_classes, config.DEVICE)

    # 3. 启动GUI
    root = tk.Tk()
    app = GaitRecognitionGUI(root, model, person_to_idx, config.DEVICE)
    root.mainloop()

