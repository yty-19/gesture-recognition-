import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
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
        self.root.title("ASL数字识别系统")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)

        # 状态变量
        self.camera_active = False
        self.cap = None
        self.inference_running = False
        self.collection_running = False
        self.current_class = 0
        self.counter = 0
        self.waiting_for_start = False

        # 创建UI
        self.create_widgets()

        # 加载模型和数据集状态
        self.model_loaded = os.path.exists(CONFIG['MODEL_FILE'])
        self.dataset_loaded = os.path.exists(CONFIG['DATASET_FILE'])

        # 更新按钮状态
        self.update_button_states()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建标签页
        tab_control = ttk.Notebook(main_frame)

        # 数据收集标签页
        self.collect_tab = ttk.Frame(tab_control)
        tab_control.add(self.collect_tab, text='数据收集')
        self.setup_collect_tab()

        # 模型训练标签页
        self.train_tab = ttk.Frame(tab_control)
        tab_control.add(self.train_tab, text='模型训练')
        self.setup_train_tab()

        # 实时识别标签页
        self.inference_tab = ttk.Frame(tab_control)
        tab_control.add(self.inference_tab, text='实时识别')
        self.setup_inference_tab()

        tab_control.pack(expand=1, fill="both")

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_collect_tab(self):
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.collect_tab, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 参数设置
        ttk.Label(control_frame, text="目标类别数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.classes_var = tk.IntVar(value=CONFIG['NUMBER_OF_CLASSES'])
        ttk.Entry(control_frame, textvariable=self.classes_var, width=10).grid(row=0, column=1, pady=2)

        ttk.Label(control_frame, text="每类样本数:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.dataset_size_var = tk.IntVar(value=CONFIG['DATASET_SIZE'])
        ttk.Entry(control_frame, textvariable=self.dataset_size_var, width=10).grid(row=1, column=1, pady=2)

        ttk.Label(control_frame, text="摄像头ID:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.camera_id_var = tk.IntVar(value=CONFIG['CAMERA_ID'])
        ttk.Entry(control_frame, textvariable=self.camera_id_var, width=10).grid(row=2, column=1, pady=2)

        # 按钮
        self.start_collect_btn = ttk.Button(control_frame, text="开始收集", command=self.start_collection)
        self.start_collect_btn.grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)

        self.stop_collect_btn = ttk.Button(control_frame, text="停止收集", command=self.stop_collection,
                                           state=tk.DISABLED)
        self.stop_collect_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.create_dirs_btn = ttk.Button(control_frame, text="创建目录", command=self.create_directories)
        self.create_dirs_btn.grid(row=5, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # 进度显示
        self.progress_var = tk.StringVar()
        self.progress_var.set("进度: 未开始")
        ttk.Label(control_frame, textvariable=self.progress_var).grid(row=6, column=0, columnspan=2, pady=10)

        self.class_var = tk.StringVar()
        self.class_var.set("当前类别: -")
        ttk.Label(control_frame, textvariable=self.class_var).grid(row=7, column=0, columnspan=2, pady=2)

        self.count_var = tk.StringVar()
        self.count_var.set("已收集: 0/0")
        ttk.Label(control_frame, textvariable=self.count_var).grid(row=8, column=0, columnspan=2, pady=2)

        # 右侧图像显示
        image_frame = ttk.LabelFrame(self.collect_tab, text="摄像头预览", padding="10")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.collect_canvas = tk.Canvas(image_frame, bg='white', width=640, height=480)
        self.collect_canvas.pack(fill=tk.BOTH, expand=True)

        # 底部日志
        log_frame = ttk.LabelFrame(self.collect_tab, text="日志", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.collect_log = scrolledtext.ScrolledText(log_frame, height=5)
        self.collect_log.pack(fill=tk.BOTH, expand=True)
        self.collect_log.config(state=tk.DISABLED)

    def setup_train_tab(self):
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.train_tab, text="训练控制", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 模型参数
        ttk.Label(control_frame, text="树的数量:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.n_estimators_var = tk.IntVar(value=100)
        ttk.Entry(control_frame, textvariable=self.n_estimators_var, width=10).grid(row=0, column=1, pady=2)

        ttk.Label(control_frame, text="最大深度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_depth_var = tk.IntVar(value=20)
        ttk.Entry(control_frame, textvariable=self.max_depth_var, width=10).grid(row=1, column=1, pady=2)

        ttk.Label(control_frame, text="测试集比例:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(control_frame, textvariable=self.test_size_var, width=10).grid(row=2, column=1, pady=2)

        # 按钮
        self.extract_features_btn = ttk.Button(control_frame, text="提取特征", command=self.extract_features)
        self.extract_features_btn.grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.EW)

        self.train_model_btn = ttk.Button(control_frame, text="训练模型", command=self.train_model)
        self.train_model_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # 数据集信息
        self.dataset_info_var = tk.StringVar()
        self.dataset_info_var.set("数据集: 未加载")
        ttk.Label(control_frame, textvariable=self.dataset_info_var).grid(row=5, column=0, columnspan=2, pady=10)

        self.model_info_var = tk.StringVar()
        self.model_info_var.set("模型: 未加载")
        ttk.Label(control_frame, textvariable=self.model_info_var).grid(row=6, column=0, columnspan=2, pady=2)

        # 右侧结果展示
        result_frame = ttk.LabelFrame(self.train_tab, text="训练结果", padding="10")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建标签页用于不同结果
        result_tabs = ttk.Notebook(result_frame)

        # 指标标签页
        metrics_tab = ttk.Frame(result_tabs)
        result_tabs.add(metrics_tab, text='性能指标')

        self.metrics_text = scrolledtext.ScrolledText(metrics_tab, height=15)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.metrics_text.config(state=tk.DISABLED)

        # 混淆矩阵标签页
        self.confusion_matrix_tab = ttk.Frame(result_tabs)
        result_tabs.add(self.confusion_matrix_tab, text='混淆矩阵')

        result_tabs.pack(expand=1, fill="both")

        # 底部日志
        log_frame = ttk.LabelFrame(self.train_tab, text="日志", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.train_log = scrolledtext.ScrolledText(log_frame, height=5)
        self.train_log.pack(fill=tk.BOTH, expand=True)
        self.train_log.config(state=tk.DISABLED)

    def setup_inference_tab(self):
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.inference_tab, text="识别控制", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 摄像头设置
        ttk.Label(control_frame, text="摄像头ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.inference_camera_id_var = tk.IntVar(value=CONFIG['CAMERA_ID'])
        ttk.Entry(control_frame, textvariable=self.inference_camera_id_var, width=10).grid(row=0, column=1, pady=2)

        # 按钮
        self.start_inference_btn = ttk.Button(control_frame, text="开始识别", command=self.start_inference)
        self.start_inference_btn.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.EW)

        self.stop_inference_btn = ttk.Button(control_frame, text="停止识别", command=self.stop_inference,
                                             state=tk.DISABLED)
        self.stop_inference_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # 识别结果
        self.result_var = tk.StringVar()
        self.result_var.set("识别结果: -")
        ttk.Label(control_frame, textvariable=self.result_var, font=("Arial", 24)).grid(
            row=3, column=0, columnspan=2, pady=20
        )

        self.confidence_var = tk.StringVar()
        self.confidence_var.set("置信度: -")
        ttk.Label(control_frame, textvariable=self.confidence_var).grid(
            row=4, column=0, columnspan=2, pady=5
        )

        # 右侧图像显示
        image_frame = ttk.LabelFrame(self.inference_tab, text="识别画面", padding="10")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.inference_canvas = tk.Canvas(image_frame, bg='white', width=640, height=480)
        self.inference_canvas.pack(fill=tk.BOTH, expand=True)

        # 底部日志
        log_frame = ttk.LabelFrame(self.inference_tab, text="日志", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.inference_log = scrolledtext.ScrolledText(log_frame, height=5)
        self.inference_log.pack(fill=tk.BOTH, expand=True)
        self.inference_log.config(state=tk.DISABLED)

    def update_button_states(self):
        # 更新按钮状态
        dataset_exists = os.path.exists(CONFIG['DATASET_FILE'])
        model_exists = os.path.exists(CONFIG['MODEL_FILE'])

        # 训练标签页按钮
        self.extract_features_btn.config(state=tk.NORMAL if os.path.exists(CONFIG['DATA_DIR']) else tk.DISABLED)
        self.train_model_btn.config(state=tk.NORMAL if dataset_exists else tk.DISABLED)

        # 识别标签页按钮
        self.start_inference_btn.config(state=tk.NORMAL if model_exists else tk.DISABLED)

        # 更新信息显示
        if dataset_exists:
            try:
                with open(CONFIG['DATASET_FILE'], 'rb') as f:
                    data_dict = pickle.load(f)
                samples = len(data_dict['data'])
                classes = len(set(data_dict['labels']))
                self.dataset_info_var.set(f"数据集: {samples}个样本, {classes}个类别")
            except:
                self.dataset_info_var.set("数据集: 已加载(读取错误)")
        else:
            self.dataset_info_var.set("数据集: 未加载")

        if model_exists:
            try:
                with open(CONFIG['MODEL_FILE'], 'rb') as f:
                    model_dict = pickle.load(f)
                model_type = type(model_dict['model']).__name__
                self.model_info_var.set(f"模型: {model_type} (已加载)")
            except:
                self.model_info_var.set("模型: 已加载(读取错误)")
        else:
            self.model_info_var.set("模型: 未加载")

    def log_message(self, text_widget, message):
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, message + "\n")
        text_widget.see(tk.END)
        text_widget.config(state=tk.DISABLED)

    def log_status(self, message):
        self.status_var.set(message)

    # ================= 数据收集功能 =================
    def create_directories(self):
        try:
            data_dir = CONFIG['DATA_DIR']
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                self.log_message(self.collect_log, f"创建主目录: {data_dir}")

            for i in range(CONFIG['NUMBER_OF_CLASSES']):
                class_dir = os.path.join(data_dir, str(i))
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                    self.log_message(self.collect_log, f"创建类别目录: {class_dir}")

            self.log_message(self.collect_log, "所有目录创建完成")
            self.log_status("目录创建完成")
            messagebox.showinfo("成功", "所有目录已成功创建")
        except Exception as e:
            self.log_message(self.collect_log, f"创建目录时出错: {str(e)}")
            self.log_status("目录创建失败")
            messagebox.showerror("错误", f"创建目录时出错: {str(e)}")

    def start_collection(self):
        if self.collection_running:
            return

        # 更新配置
        CONFIG['NUMBER_OF_CLASSES'] = self.classes_var.get()
        CONFIG['DATASET_SIZE'] = self.dataset_size_var.get()
        CONFIG['CAMERA_ID'] = self.camera_id_var.get()

        # 创建目录
        self.create_directories()

        # 启动收集线程
        self.collection_running = True
        self.start_collect_btn.config(state=tk.DISABLED)
        self.stop_collect_btn.config(state=tk.NORMAL)
        self.create_dirs_btn.config(state=tk.DISABLED)

        self.current_class = 0
        self.counter = 0
        self.waiting_for_start = True

        self.log_message(self.collect_log, "开始数据收集...")
        self.log_status("数据收集中...")

        # 启动摄像头
        self.cap = cv2.VideoCapture(CONFIG['CAMERA_ID'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['TARGET_WIDTH'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['TARGET_HEIGHT'])

        if not self.cap.isOpened():
            self.log_message(self.collect_log, "错误: 无法打开摄像头")
            self.stop_collection()
            return

        # 更新状态
        self.progress_var.set(f"进度: 0/{CONFIG['NUMBER_OF_CLASSES']} 类别")
        self.class_var.set(f"当前类别: 准备中...")
        self.count_var.set(f"已收集: 0/{CONFIG['DATASET_SIZE']}")

        # 开始收集循环
        self.collect_next_class()

    def collect_next_class(self):
        if not self.collection_running or self.current_class >= CONFIG['NUMBER_OF_CLASSES']:
            self.stop_collection()
            return

        self.class_var.set(f"当前类别: {self.current_class}")
        self.log_message(self.collect_log, f"开始收集类别 {self.current_class} 的数据")
        self.waiting_for_start = True

        # 显示准备信息
        self.show_preparation_screen()

    def show_preparation_screen(self):
        if not self.collection_running or not self.waiting_for_start:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.show_preparation_screen)
            return

        # 使用PIL添加中文文本（解决乱码问题）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)

        try:
            # 尝试加载中文字体
            font = ImageFont.truetype("simhei.ttf", 40)
        except:
            # 如果失败，使用默认字体（可能不支持中文）
            font = ImageFont.load_default()

        # 添加准备文本
        draw.text((50, 50), f'准备收集数字 {self.current_class} 的图像', font=font, fill=(250, 0, 0))
        draw.text((50, 100), '准备好后按空格键开始收集', font=font, fill=(250, 0, 0))

        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 显示图像
        self.display_collect_frame(frame)
        self.root.after(10, self.show_preparation_screen)

    def start_class_collection(self):
        self.waiting_for_start = False
        self.counter = 0
        self.log_message(self.collect_log, f"开始收集类别 {self.current_class} 的图像")
        self.collect_images()

    def collect_images(self):
        if not self.collection_running or self.counter >= CONFIG['DATASET_SIZE']:
            # 完成当前类别，转到下一类别
            self.current_class += 1
            self.progress_var.set(f"进度: {self.current_class}/{CONFIG['NUMBER_OF_CLASSES']} 类别")
            self.root.after(10, self.collect_next_class)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.collect_images)
            return

        # 使用PIL添加计数文本（解决乱码问题）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)

        try:
            # 尝试加载中文字体
            font = ImageFont.truetype("simhei.ttf", 40)
        except:
            # 如果失败，使用默认字体（可能不支持中文）
            font = ImageFont.load_default()

        # 添加计数文本
        draw.text((50, 50), f'采集: {self.counter}/{CONFIG["DATASET_SIZE"]}', font=font, fill=(250, 0, 0))

        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 显示图像
        self.display_collect_frame(frame)

        # 保存图像
        img_path = os.path.join(CONFIG['DATA_DIR'], str(self.current_class), f'{self.counter}.jpg')
        cv2.imwrite(img_path, frame)
        self.counter += 1
        self.count_var.set(f"已收集: {self.counter}/{CONFIG['DATASET_SIZE']}")

        self.root.after(25, self.collect_images)

    def display_collect_frame(self, frame):
        # 调整大小以适应画布
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # 使用兼容的方式处理图像重采样
        try:
            # 尝试使用新版本的Resampling枚举
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        except AttributeError:
            # 回退到旧版本的常量
            img = img.resize((640, 480), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img)

        self.collect_canvas.img_tk = img_tk  # 保持引用
        self.collect_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def stop_collection(self):
        self.collection_running = False
        self.waiting_for_start = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_collect_btn.config(state=tk.NORMAL)
        self.stop_collect_btn.config(state=tk.DISABLED)
        self.create_dirs_btn.config(state=tk.NORMAL)

        self.log_message(self.collect_log, "数据收集已停止")
        self.log_status("数据收集停止")

        self.progress_var.set("进度: 已停止")
        self.class_var.set("当前类别: -")
        self.count_var.set("已收集: 0/0")

        # 清除画布
        self.collect_canvas.delete("all")
        self.collect_canvas.create_text(320, 240, text="摄像头已关闭", font=("Arial", 24), fill="black")

    # ================= 特征提取功能 =================
    def extract_features(self):
        def extraction_thread():
            self.log_message(self.train_log, "开始特征提取...")
            self.log_status("特征提取中...")

            try:
                # 初始化MediaPipe
                hands = mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.5
                )

                data = []
                labels = []
                total_images = sum(len(os.listdir(os.path.join(CONFIG['DATA_DIR'], dir_)))
                                   for dir_ in os.listdir(CONFIG['DATA_DIR']))

                processed = 0
                self.log_message(self.train_log, f"总共需要处理 {total_images} 张图像")

                for dir_ in sorted(os.listdir(CONFIG['DATA_DIR'])):
                    class_dir = os.path.join(CONFIG['DATA_DIR'], dir_)
                    if not os.path.isdir(class_dir):
                        continue

                    for img_file in os.listdir(class_dir):
                        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            continue

                        img_path = os.path.join(class_dir, img_file)
                        img = cv2.imread(img_path)
                        if img is None:
                            self.log_message(self.train_log, f"无法读取图像: {img_path}")
                            continue

                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = hands.process(img_rgb)

                        if results.multi_hand_landmarks:
                            hand_landmarks = results.multi_hand_landmarks[0]
                            data_aux = []
                            x_ = []
                            y_ = []

                            for landmark in hand_landmarks.landmark:
                                x_.append(landmark.x)
                                y_.append(landmark.y)

                            for landmark in hand_landmarks.landmark:
                                data_aux.append(landmark.x - min(x_))
                                data_aux.append(landmark.y - min(y_))

                            data.append(data_aux)
                            labels.append(dir_)

                        processed += 1
                        if processed % 50 == 0:
                            self.log_message(self.train_log, f"已处理 {processed}/{total_images} 张图像")

                # 保存数据集
                with open(CONFIG['DATASET_FILE'], 'wb') as f:
                    pickle.dump({'data': data, 'labels': labels}, f)

                self.log_message(self.train_log, f"特征提取完成! 保存了 {len(data)} 个样本")
                self.log_status("特征提取完成")
                messagebox.showinfo("成功", f"特征提取完成! 保存了 {len(data)} 个样本")

                # 更新状态
                self.dataset_loaded = True
                self.update_button_states()

            except Exception as e:
                self.log_message(self.train_log, f"特征提取出错: {str(e)}")
                self.log_status("特征提取失败")
                messagebox.showerror("错误", f"特征提取出错: {str(e)}")

        # 启动提取线程
        threading.Thread(target=extraction_thread, daemon=True).start()

    # ================= 模型训练功能 =================
    def train_model(self):
        def training_thread():
            try:
                self.log_message(self.train_log, "开始模型训练...")
                self.log_status("模型训练中...")

                # 加载数据
                with open(CONFIG['DATASET_FILE'], 'rb') as f:
                    data_dict = pickle.load(f)

                X = np.asarray(data_dict['data'])
                y = np.asarray(data_dict['labels'])

                # 分割数据集
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size_var.get(), random_state=42, stratify=y
                )

                # 初始化模型
                model = RandomForestClassifier(
                    n_estimators=self.n_estimators_var.get(),
                    max_depth=self.max_depth_var.get(),
                    random_state=42
                )

                # 训练模型
                model.fit(X_train, y_train)

                # 评估模型
                y_pred = model.predict(X_test)
                train_score = model.score(X_train, y_train)
                test_score = accuracy_score(y_test, y_pred)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)

                # 保存模型
                with open(CONFIG['MODEL_FILE'], 'wb') as f:
                    pickle.dump({'model': model}, f)

                # 显示结果
                report = (
                    f"模型训练完成!\n"
                    f"训练准确率: {train_score * 100:.2f}%\n"
                    f"测试准确率: {test_score * 100:.2f}%\n"
                    f"交叉验证准确率: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 2 * 100:.2f}%)\n\n"
                    f"分类报告:\n{classification_report(y_test, y_pred)}"
                )

                self.log_message(self.train_log, report)
                self.log_status("模型训练完成")

                # 更新指标显示
                self.metrics_text.config(state=tk.NORMAL)
                self.metrics_text.delete(1.0, tk.END)
                self.metrics_text.insert(tk.END, report)
                self.metrics_text.config(state=tk.DISABLED)

                # 绘制混淆矩阵
                self.plot_confusion_matrix(y_test, y_pred)

                # 更新状态
                self.model_loaded = True
                self.update_button_states()

                messagebox.showinfo("成功", "模型训练完成!")

            except Exception as e:
                self.log_message(self.train_log, f"模型训练出错: {str(e)}")
                self.log_status("模型训练失败")
                messagebox.showerror("错误", f"模型训练出错: {str(e)}")

        # 启动训练线程
        threading.Thread(target=training_thread, daemon=True).start()

    def plot_confusion_matrix(self, y_test, y_pred):
        # 清除之前的图形
        for widget in self.confusion_matrix_tab.winfo_children():
            widget.destroy()

        # 创建新图形
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.confusion_matrix_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ================= 实时识别功能 =================
    def start_inference(self):
        if self.inference_running:
            return

        self.inference_running = True
        self.start_inference_btn.config(state=tk.DISABLED)
        self.stop_inference_btn.config(state=tk.NORMAL)

        # 加载模型
        try:
            with open(CONFIG['MODEL_FILE'], 'rb') as f:
                model_dict = pickle.load(f)
            self.model = model_dict['model']
            self.labels_dict = {str(i): str(i) for i in range(10)}

            self.log_message(self.inference_log, "模型加载成功")
        except Exception as e:
            self.log_message(self.inference_log, f"加载模型失败: {str(e)}")
            self.stop_inference()
            return

        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.inference_camera_id_var.get())
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['TARGET_WIDTH'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['TARGET_HEIGHT'])

        if not self.cap.isOpened():
            self.log_message(self.inference_log, "无法打开摄像头")
            self.stop_inference()
            return

        # 初始化MediaPipe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.log_message(self.inference_log, "开始实时识别...")
        self.log_status("实时识别中...")

        # 开始识别循环
        self.run_inference()

    def run_inference(self):
        if not self.inference_running or not self.cap.isOpened():
            self.stop_inference()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.run_inference)
            return

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles(color=(0, 255, 0), thickness=2),
                    mp_drawing_styles(color=(255, 0, 0), thickness=2)
                )

                # 处理关键点
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # 归一化坐标
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # 绘制边界框
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # 进行预测
                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_number = self.labels_dict.get(str(prediction[0]), '?')

                # 获取置信度
                probabilities = self.model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(probabilities) * 100

                # 更新UI
                self.result_var.set(f"识别结果: {predicted_number}")
                self.confidence_var.set(f"置信度: {confidence:.2f}%")

                # 在图像上绘制结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Number: {predicted_number}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # 显示图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # 使用兼容的方式处理图像重采样
        try:
            # 尝试使用新版本的Resampling枚举
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        except AttributeError:
            # 回退到旧版本的常量
            img = img.resize((640, 480), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img)

        self.inference_canvas.img_tk = img_tk  # 保持引用
        self.inference_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        self.root.after(10, self.run_inference)

    def stop_inference(self):
        self.inference_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_inference_btn.config(state=tk.NORMAL)
        self.stop_inference_btn.config(state=tk.DISABLED)

        self.log_message(self.inference_log, "实时识别已停止")
        self.log_status("实时识别停止")

        # 清除画布
        self.inference_canvas.delete("all")
        self.inference_canvas.create_text(320, 240, text="摄像头已关闭", font=("Arial", 24), fill="black")

        self.result_var.set("识别结果: -")
        self.confidence_var.set("置信度: -")

    # 处理键盘事件
    def handle_keypress(self, event):
        if not self.collection_running:
            return

        # 在准备阶段按空格键开始收集
        if event.keysym == 'space' and self.waiting_for_start:
            self.start_class_collection()


if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)

    # 绑定键盘事件
    root.bind('<Key>', app.handle_keypress)

    root.mainloop()