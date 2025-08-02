import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox, scrolledtext
import threading
import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import shutil

# 全局配置
CONFIG = {
    'DATA_DIR': './data',
    'MODEL_FILE': './model.p',
    'DATASET_FILE': './data.pickle',
    'NUMBER_OF_CLASSES': 10,
    'DATASET_SIZE': 100,
    'CAMERA_ID': 0,
    'TARGET_WIDTH': 1280,
    'TARGET_HEIGHT': 720
}

# 初始化MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec


class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL数字识别系统 - 专业版")
        self.root.geometry("1280x820")
        self.root.minsize(1200, 800)

        # 状态变量
        self.cap = None
        self.inference_running = False
        self.collection_running = False
        self.waiting_for_start = False
        self.model = None
        self.hands = None
        self.labels_dict = {}

        self.create_widgets()
        self.update_button_states()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tab_control = ttk.Notebook(main_frame, bootstyle="primary")
        self.collect_tab = ttk.Frame(tab_control, padding=10)
        tab_control.add(self.collect_tab, text='  数据收集  ')
        self.setup_collect_tab()

        self.train_tab = ttk.Frame(tab_control, padding=10)
        tab_control.add(self.train_tab, text='  模型训练  ')
        self.setup_train_tab()

        self.inference_tab = ttk.Frame(tab_control, padding=10)
        tab_control.add(self.inference_tab, text='  实时识别  ')
        self.setup_inference_tab()
        tab_control.pack(expand=1, fill="both")

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(self.root, textvariable=self.status_var, padding=(10, 5), anchor=W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_collect_tab(self):
        self.collect_tab.columnconfigure(1, weight=3)
        self.collect_tab.rowconfigure(0, weight=3)
        self.collect_tab.rowconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(self.collect_tab, text=" 控制面板 ", padding=15)
        control_frame.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 15))

        param_group = ttk.Frame(control_frame)
        param_group.pack(fill=X, pady=5)
        param_group.columnconfigure(1, weight=1)

        ttk.Label(param_group, text="目标类别数:").grid(row=0, column=0, sticky=W, pady=4)
        self.classes_var = tk.IntVar(value=CONFIG['NUMBER_OF_CLASSES'])
        ttk.Entry(param_group, textvariable=self.classes_var, width=8, bootstyle="primary").grid(row=0, column=1,
                                                                                                 sticky=E, pady=4)

        ttk.Label(param_group, text="每类样本数:").grid(row=1, column=0, sticky=W, pady=4)
        self.dataset_size_var = tk.IntVar(value=CONFIG['DATASET_SIZE'])
        ttk.Entry(param_group, textvariable=self.dataset_size_var, width=8, bootstyle="primary").grid(row=1, column=1,
                                                                                                      sticky=E, pady=4)

        ttk.Label(param_group, text="摄像头ID:").grid(row=2, column=0, sticky=W, pady=4)
        self.camera_id_var = tk.IntVar(value=CONFIG['CAMERA_ID'])
        ttk.Entry(param_group, textvariable=self.camera_id_var, width=8, bootstyle="primary").grid(row=2, column=1,
                                                                                                   sticky=E, pady=4)

        self.start_collect_btn = ttk.Button(control_frame, text="开始收集", command=self.start_collection,
                                            bootstyle="success")
        self.start_collect_btn.pack(fill=X, pady=(20, 5))

        self.stop_collect_btn = ttk.Button(control_frame, text="停止收集", command=self.stop_collection,
                                           state=tk.DISABLED, bootstyle="danger-outline")
        self.stop_collect_btn.pack(fill=X, pady=5)

        ttk.Separator(control_frame).pack(fill=X, pady=10, padx=5)
        self.reset_btn = ttk.Button(control_frame, text="一键重置项目", command=self.reset_project, bootstyle="danger")
        self.reset_btn.pack(fill=X, pady=5)

        ttk.Separator(control_frame).pack(fill=X, pady=10, padx=5)

        self.progress_var = tk.StringVar(value="进度: 未开始")
        ttk.Label(control_frame, textvariable=self.progress_var, font="-size 10").pack(fill=X, pady=2, anchor=W)
        self.class_var = tk.StringVar(value="当前类别: -")
        ttk.Label(control_frame, textvariable=self.class_var, font="-size 10").pack(fill=X, pady=2, anchor=W)
        self.count_var = tk.StringVar(value="已收集: 0/0")
        ttk.Label(control_frame, textvariable=self.count_var, font="-size 10").pack(fill=X, pady=2, anchor=W)

        image_frame = ttk.LabelFrame(self.collect_tab, text=" 摄像头预览 ", padding=10)
        image_frame.grid(row=0, column=1, sticky="nswe")
        self.collect_canvas = tk.Canvas(image_frame, bg='black')
        self.collect_canvas.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.LabelFrame(self.collect_tab, text=" 日志 ", padding=10)
        log_frame.grid(row=1, column=1, sticky="nswe", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.collect_log = scrolledtext.ScrolledText(log_frame, height=6, state=tk.DISABLED)
        self.collect_log.grid(row=0, column=0, sticky="nswe")

    def setup_train_tab(self):
        self.train_tab.columnconfigure(1, weight=3)
        self.train_tab.rowconfigure(0, weight=1)
        self.train_tab.rowconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(self.train_tab, text=" 训练控制 ", padding=15)
        control_frame.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 15))
        control_frame.columnconfigure(0, weight=1)

        param_group = ttk.Frame(control_frame)
        param_group.pack(fill=X, pady=5)
        param_group.columnconfigure(1, weight=1)

        ttk.Label(param_group, text="树的数量:").grid(row=0, column=0, sticky=W, pady=4)
        self.n_estimators_var = tk.IntVar(value=100)
        ttk.Entry(param_group, textvariable=self.n_estimators_var, width=8, bootstyle="primary").grid(row=0, column=1,
                                                                                                      sticky=E, pady=4)
        ttk.Label(param_group, text="最大深度:").grid(row=1, column=0, sticky=W, pady=4)
        self.max_depth_var = tk.IntVar(value=20)
        ttk.Entry(param_group, textvariable=self.max_depth_var, width=8, bootstyle="primary").grid(row=1, column=1,
                                                                                                   sticky=E, pady=4)
        ttk.Label(param_group, text="测试集比例:").grid(row=2, column=0, sticky=W, pady=4)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(param_group, textvariable=self.test_size_var, width=8, bootstyle="primary").grid(row=2, column=1,
                                                                                                   sticky=E, pady=4)

        self.extract_features_btn = ttk.Button(control_frame, text="提取特征", command=self.extract_features,
                                               bootstyle="primary")
        self.extract_features_btn.pack(fill=X, pady=(20, 5))
        self.train_model_btn = ttk.Button(control_frame, text="训练模型", command=self.train_model, bootstyle="success")
        self.train_model_btn.pack(fill=X, pady=5)

        ttk.Separator(control_frame).pack(fill=X, pady=20)
        self.dataset_info_var = tk.StringVar(value="数据集: 未加载")
        ttk.Label(control_frame, textvariable=self.dataset_info_var).pack(fill=X, pady=2, anchor=W)
        self.model_info_var = tk.StringVar(value="模型: 未加载")
        ttk.Label(control_frame, textvariable=self.model_info_var).pack(fill=X, pady=2, anchor=W)

        result_frame = ttk.LabelFrame(self.train_tab, text="训练结果", padding=10)
        result_frame.grid(row=0, column=1, sticky="nswe")
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_tabs = ttk.Notebook(result_frame, bootstyle="primary")
        result_tabs.pack(expand=1, fill="both")
        metrics_tab = ttk.Frame(result_tabs, padding=10)
        result_tabs.add(metrics_tab, text='性能指标')
        self.metrics_text = scrolledtext.ScrolledText(metrics_tab, height=15, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.confusion_matrix_tab = ttk.Frame(result_tabs)
        result_tabs.add(self.confusion_matrix_tab, text='混淆矩阵')

        log_frame = ttk.LabelFrame(self.train_tab, text=" 日志 ", padding=10)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nswe", pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.train_log = scrolledtext.ScrolledText(log_frame, height=6, state=tk.DISABLED)
        self.train_log.grid(row=0, column=0, sticky="nswe")

    def setup_inference_tab(self):
        self.inference_tab.columnconfigure(1, weight=3)
        self.inference_tab.rowconfigure(0, weight=1)
        control_frame = ttk.LabelFrame(self.inference_tab, text=" 识别控制 ", padding=15)
        control_frame.grid(row=0, column=0, sticky="ns", padx=(0, 15))
        self.start_inference_btn = ttk.Button(control_frame, text="开始识别", command=self.start_inference,
                                              bootstyle="success")
        self.start_inference_btn.pack(fill=X, pady=10, ipady=5)
        self.stop_inference_btn = ttk.Button(control_frame, text="停止识别", command=self.stop_inference,
                                             state=tk.DISABLED, bootstyle="danger-outline")
        self.stop_inference_btn.pack(fill=X, pady=5, ipady=5)
        ttk.Separator(control_frame).pack(fill=X, pady=20)
        self.result_var = tk.StringVar(value="结果: -")
        ttk.Label(control_frame, textvariable=self.result_var, font="-size 24 -weight bold").pack(pady=10)
        self.confidence_var = tk.StringVar(value="置信度: -")
        ttk.Label(control_frame, textvariable=self.confidence_var, font="-size 12").pack(pady=5)
        image_frame = ttk.LabelFrame(self.inference_tab, text="识别画面", padding=10)
        image_frame.grid(row=0, column=1, sticky="nswe")
        self.inference_canvas = tk.Canvas(image_frame, bg='black')
        self.inference_canvas.pack(fill=tk.BOTH, expand=True)

    def reset_project(self):
        if messagebox.askyesno("确认重置",
                               "此操作将永久删除所有已收集的数据、提取的特征和训练好的模型！\n\n您确定要继续吗？",
                               icon='warning'):
            threading.Thread(target=self._reset_project_thread, daemon=True).start()

    def _reset_project_thread(self):
        self.log_status("正在重置项目...")
        try:
            if self.collection_running: self.stop_collection()
            if self.inference_running: self.stop_inference()

            if os.path.exists(CONFIG['DATA_DIR']):
                shutil.rmtree(CONFIG['DATA_DIR'])
                self.log_message(self.collect_log, f"目录 {CONFIG['DATA_DIR']} 已删除。")

            for file_path in [CONFIG['DATASET_FILE'], CONFIG['MODEL_FILE']]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.log_message(self.collect_log, f"文件 {file_path} 已删除。")

            self.log_status("项目已重置！")
            messagebox.showinfo("成功", "项目已成功重置！")

        except Exception as e:
            self.log_status("重置失败！")
            messagebox.showerror("错误", f"重置项目时发生错误: {e}")
        finally:
            self.update_button_states()

    def update_button_states(self):
        dataset_exists = os.path.exists(CONFIG['DATASET_FILE'])
        model_exists = os.path.exists(CONFIG['MODEL_FILE'])
        data_dir_exists = os.path.exists(CONFIG['DATA_DIR'])

        log_widget = self.train_log if hasattr(self, 'train_log') else self.collect_log
        self.log_message(log_widget, f"状态更新检查: data目录是否存在 -> {data_dir_exists}")

        self.extract_features_btn.config(state=tk.NORMAL if data_dir_exists else tk.DISABLED)
        self.train_model_btn.config(state=tk.NORMAL if dataset_exists else tk.DISABLED)
        self.start_inference_btn.config(state=tk.NORMAL if model_exists else tk.DISABLED)

        if dataset_exists:
            try:
                with open(CONFIG['DATASET_FILE'], 'rb') as f:
                    data_dict = pickle.load(f)
                samples = len(data_dict['data'])
                classes = len(set(data_dict['labels']))
                self.dataset_info_var.set(f"数据集: {samples}个样本, {classes}个类别")
            except Exception:
                self.dataset_info_var.set("数据集: 已加载(读取错误)")
        else:
            self.dataset_info_var.set("数据集: 未加载")

        if model_exists:
            try:
                with open(CONFIG['MODEL_FILE'], 'rb') as f:
                    model_dict = pickle.load(f)
                model_type = type(model_dict['model']).__name__
                self.model_info_var.set(f"模型: {model_type} (已加载)")
            except Exception:
                self.model_info_var.set("模型: 已加载(读取错误)")
        else:
            self.model_info_var.set("模型: 未加载")

    def log_message(self, text_widget, message):
        if not hasattr(text_widget, 'winfo_exists') or not text_widget.winfo_exists():
            return
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, message + "\n")
        text_widget.see(tk.END)
        text_widget.config(state=tk.DISABLED)

    def log_status(self, message):
        self.status_var.set(message)

    def _ensure_directories_exist(self):
        """检查并创建所需的数据目录，成功返回True，失败返回False。"""
        try:
            data_dir = CONFIG['DATA_DIR']
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                self.log_message(self.collect_log, f"主数据目录 '{data_dir}' 已创建。")

            num_classes = self.classes_var.get()
            for i in range(num_classes):
                class_dir = os.path.join(data_dir, str(i))
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

            self.log_message(self.collect_log, "数据目录检查完毕，准备就绪。")
            self.update_button_states()
            return True
        except Exception as e:
            messagebox.showerror("目录错误", f"创建数据目录时出错: {str(e)}")
            self.log_status("错误：无法创建目录！")
            return False

    def start_collection(self):
        if self.collection_running: return

        if not self._ensure_directories_exist():
            return

        self.collection_running = True
        self.start_collect_btn.config(state=tk.DISABLED)
        self.stop_collect_btn.config(state=tk.NORMAL)

        self.cap = cv2.VideoCapture(self.camera_id_var.get())
        if not self.cap.isOpened():
            messagebox.showerror("摄像头错误", "无法打开摄像头，请检查摄像头ID或连接。")
            self.stop_collection()
            return

        self.current_class = 0
        self.counter = 0
        self.collect_next_class()

    def collect_next_class(self):
        if not self.collection_running or self.current_class >= self.classes_var.get():
            was_running = self.collection_running
            self.stop_collection()  # 先停止
            if was_running:
                messagebox.showinfo("完成", "所有类别数据收集完毕！")  # 再弹窗
            return

        self.class_var.set(f"当前类别: {self.current_class}")
        self.progress_var.set(f"进度: {self.current_class}/{self.classes_var.get()} 类别")
        self.waiting_for_start = True
        self.show_preparation_screen()

    def show_preparation_screen(self):
        if not self.waiting_for_start: return
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(self.collect_canvas, frame,
                               f'准备收集: 数字 {self.current_class}\n\n请将手势摆好后按 <空格键> 开始')
        self.root.after(10, self.show_preparation_screen)

    def start_class_collection(self):
        self.waiting_for_start = False
        self.counter = 0
        self.collect_images()

    def collect_images(self):
        if not self.collection_running or self.counter >= self.dataset_size_var.get():
            self.current_class += 1
            self.collect_next_class()
            return
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(self.collect_canvas, frame, f'采集中: {self.counter + 1}/{self.dataset_size_var.get()}')
            img_path = os.path.join(CONFIG['DATA_DIR'], str(self.current_class), f'{self.counter}.jpg')
            cv2.imwrite(img_path, frame)
            self.counter += 1
            self.count_var.set(f"已收集: {self.counter}/{self.dataset_size_var.get()}")
        self.root.after(30, self.collect_images)

    def stop_collection(self):
        self.collection_running = False
        self.waiting_for_start = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_collect_btn.config(state=tk.NORMAL)
        self.stop_collect_btn.config(state=tk.DISABLED)
        self.display_frame(self.collect_canvas, None, "摄像头已关闭")
        self.progress_var.set("进度: 已停止")
        self.update_button_states()

    def display_frame(self, canvas, frame, text=None):
        self.root.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        canvas.delete("all")

        if frame is None:
            if canvas_width > 1 and canvas_height > 1:
                canvas.create_text(canvas_width / 2, canvas_height / 2, text=text, font=("-size", 20, "bold"),
                                   fill="grey")
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        if text:
            draw = ImageDraw.Draw(img_pil, "RGBA")
            try:
                font = ImageFont.truetype("simhei.ttf", 40)
            except IOError:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((0, 0), text, font=font, align="center")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (img_pil.width - text_width) / 2
            text_y = 50

            draw.rectangle([(text_x - 20, text_y - 10), (text_x + text_width + 20, text_y + text_height + 20)],
                           fill=(0, 0, 0, 128))
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255), align="center")

        if canvas_width <= 1 or canvas_height <= 1: return

        img_aspect = img_pil.width / img_pil.height
        canvas_aspect = canvas_width / canvas_height

        if img_aspect > canvas_aspect:
            new_width = canvas_width
            new_height = int(new_width / img_aspect)
        else:
            new_height = canvas_height
            new_width = int(new_height * img_aspect)

        try:
            img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError:
            img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img_resized)
        canvas.img_tk = img_tk
        canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=img_tk)

    def extract_features(self):
        threading.Thread(target=self._extract_features_thread, daemon=True).start()

    def _extract_features_thread(self):
        self.log_message(self.train_log, "开始特征提取...")
        self.log_status("特征提取中...")
        try:
            hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
            data, labels = [], []
            total_images = sum(len(files) for _, _, files in os.walk(CONFIG['DATA_DIR']))
            processed = 0
            self.log_message(self.train_log, f"总共需要处理 {total_images} 张图像")

            for dir_ in sorted(os.listdir(CONFIG['DATA_DIR'])):
                class_dir = os.path.join(CONFIG['DATA_DIR'], dir_)
                if not os.path.isdir(class_dir): continue
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is None: continue
                    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        data_aux, x_, y_ = [], [], []
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)
                        for lm in hand_landmarks.landmark:
                            data_aux.append(lm.x - min(x_))
                            data_aux.append(lm.y - min(y_))
                        data.append(data_aux)
                        labels.append(dir_)
                    processed += 1
                    if processed % 100 == 0:
                        self.log_message(self.train_log, f"已处理 {processed}/{total_images} 张图像...")

            with open(CONFIG['DATASET_FILE'], 'wb') as f:
                pickle.dump({'data': data, 'labels': labels}, f)
            self.log_message(self.train_log, f"特征提取完成! 保存了 {len(data)} 个样本。")
            self.log_status("特征提取完成")
            messagebox.showinfo("成功", f"特征提取完成! 保存了 {len(data)} 个样本。")
            self.update_button_states()
        except Exception as e:
            messagebox.showerror("错误", f"特征提取出错: {e}")
            self.log_status("特征提取失败")

    def train_model(self):
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        self.log_message(self.train_log, "开始模型训练...")
        self.log_status("模型训练中...")
        try:
            with open(CONFIG['DATASET_FILE'], 'rb') as f:
                data_dict = pickle.load(f)
            X, y = np.asarray(data_dict['data']), np.asarray(data_dict['labels'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size_var.get(),
                                                                random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=self.n_estimators_var.get(), max_depth=self.max_depth_var.get(),
                                           random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, f"测试准确率: {accuracy_score(y_test, y_pred) * 100:.2f}%\n\n{report}")
            self.metrics_text.config(state=tk.DISABLED)
            self.plot_confusion_matrix(y_test, y_pred)
            with open(CONFIG['MODEL_FILE'], 'wb') as f:
                pickle.dump({'model': model}, f)
            self.log_message(self.train_log, "模型训练完成！")
            self.log_status("模型训练完成")
            messagebox.showinfo("成功", "模型训练完成!")
            self.update_button_states()
        except Exception as e:
            messagebox.showerror("错误", f"模型训练出错: {e}")
            self.log_status("模型训练失败")

    def plot_confusion_matrix(self, y_test, y_pred):
        for widget in self.confusion_matrix_tab.winfo_children():
            widget.destroy()
        labels = sorted(list(set(y_test)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        canvas = FigureCanvasTkAgg(fig, master=self.confusion_matrix_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def start_inference(self):
        if self.inference_running: return
        self.inference_running = True
        self.start_inference_btn.config(state=tk.DISABLED)
        self.stop_inference_btn.config(state=tk.NORMAL)
        try:
            with open(CONFIG['MODEL_FILE'], 'rb') as f:
                self.model = pickle.load(f)['model']
            self.labels_dict = {i: str(i) for i in range(self.classes_var.get())}
            self.cap = cv2.VideoCapture(self.camera_id_var.get())
            self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                                        min_tracking_confidence=0.5)
            self.log_status("实时识别中...")
            self.run_inference()
        except Exception as e:
            messagebox.showerror("错误", f"启动识别失败: {e}")
            self.stop_inference()

    def run_inference(self):
        if not self.inference_running: return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.run_inference)
            return

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                data_aux, x_, y_ = [], [], []
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_char = self.labels_dict.get(int(prediction[0]), '?')
                probabilities = self.model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(probabilities) * 100

                self.result_var.set(f"结果: {predicted_char}")
                self.confidence_var.set(f"置信度: {confidence:.2f}%")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted_char} ({confidence:.1f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)
        else:
            self.result_var.set("结果: -")
            self.confidence_var.set("置信度: -")

        self.display_frame(self.inference_canvas, frame)
        self.root.after(10, self.run_inference)

    def stop_inference(self):
        self.inference_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_inference_btn.config(state=tk.NORMAL)
        self.stop_inference_btn.config(state=tk.DISABLED)
        self.display_frame(self.inference_canvas, None, "摄像头已关闭")
        self.log_status("实时识别停止")
        self.result_var.set("识别结果: -")
        self.confidence_var.set("置信度: -")

    def handle_keypress(self, event):
        if event.keysym == 'space' and self.waiting_for_start:
            self.start_class_collection()


if __name__ == "__main__":
    root = ttk.Window(themename="litera")
    app = ASLApp(root)
    root.bind('<Key>', app.handle_keypress)
    root.mainloop()