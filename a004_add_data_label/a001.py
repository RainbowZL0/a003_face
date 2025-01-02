import os
import tkinter as tk
from collections import defaultdict
from tkinter import filedialog, messagebox


class ImageRenamer:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("人脸图片批量重命名工具")
        self.window.geometry("600x400")

        # 创建界面元素
        self.create_widgets()

        # 存储选择的目录路径
        self.folder_path = ""

        # 存储已使用的人员编号
        self.current_person_number = 1

    def create_widgets(self):
        # 选择文件夹按钮
        self.select_button = tk.Button(
            self.window,
            text="选择图片文件夹",
            command=self.select_folder
        )
        self.select_button.pack(pady=10)

        # 显示选择的路径
        self.path_label = tk.Label(self.window, text="未选择文件夹")
        self.path_label.pack(pady=5)

        # 模态选择
        self.modality_frame = tk.LabelFrame(self.window, text="图片模态")
        self.modality_frame.pack(pady=10)

        self.modality_var = tk.StringVar(value="vis")
        tk.Radiobutton(
            self.modality_frame,
            text="可见光(vis)",
            variable=self.modality_var,
            value="vis"
        ).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(
            self.modality_frame,
            text="红外(infrared)",
            variable=self.modality_var,
            value="infrared"
        ).pack(side=tk.LEFT, padx=10)

        # 开始重命名按钮
        self.rename_button = tk.Button(
            self.window,
            text="开始重命名",
            command=self.start_rename,
            state=tk.DISABLED
        )
        self.rename_button.pack(pady=10)

        # 日志显示区域
        self.log_text = tk.Text(self.window, height=10, width=50)
        self.log_text.pack(pady=10)

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.path_label.config(text=f"已选择: {self.folder_path}")
            self.rename_button.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, "已选择文件夹，可以开始重命名...\n")

    def is_image_file(self, filename):
        """检查文件是否为图片"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        return os.path.splitext(filename.lower())[1] in image_extensions

    def get_person_images(self):
        """获取所有图片并按人分组"""
        # 获取目录下所有图片文件
        image_files = [f for f in os.listdir(self.folder_path)
                       if self.is_image_file(f)]

        # 用于存储每个人的图片列表
        person_images = defaultdict(list)

        # 遍历所有图片，尝试从现有文件名中提取人员信息
        for image_file in image_files:
            # 这里需要根据实际的文件命名规则来提取人员信息
            # 假设文件名中包含人员标识信息
            person_images[self.current_person_number].append(image_file)
            self.current_person_number += 1

        return person_images

    def start_rename(self):
        if not self.folder_path:
            messagebox.showerror("错误", "请先选择文件夹！")
            return

        try:
            # 获取按人分组的图片
            person_images = self.get_person_images()

            # 获取选择的模态
            modality = self.modality_var.get()

            # 记录重命名操作
            renamed_count = 0

            # 对每个人的图片进行重命名
            for person_num, images in person_images.items():
                image_num = 1
                for old_name in images:
                    # 构建新文件名
                    new_name = f"{person_num:06d}_{modality}_{image_num:06d}{os.path.splitext(old_name)[1]}"

                    # 完整的文件路径
                    old_path = os.path.join(self.folder_path, old_name)
                    new_path = os.path.join(self.folder_path, new_name)

                    # 重命名文件
                    os.rename(old_path, new_path)

                    # 更新日志
                    log_msg = f"重命名: {old_name} -> {new_name}\n"
                    self.log_text.insert(tk.END, log_msg)
                    self.log_text.see(tk.END)

                    renamed_count += 1
                    image_num += 1

            # 完成提示
            messagebox.showinfo("完成", f"重命名完成！共处理 {renamed_count} 个文件。")

        except Exception as e:
            messagebox.showerror("错误", f"重命名过程中出现错误：{str(e)}")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = ImageRenamer()
    app.run()
