import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import shutil
from datetime import datetime

CAMERA_ID = 0
ADAPTIVE_DATA_DIR = './adaptive_data'  # 存储自适应学习数据的目录
MODEL_FILE = './model.p'
ADAPTIVE_MODEL_FILE = './model_adaptive.p'

# 创建自适应数据目录
if not os.path.exists(ADAPTIVE_DATA_DIR):
    os.makedirs(ADAPTIVE_DATA_DIR)
    for i in range(10):
        os.makedirs(os.path.join(ADAPTIVE_DATA_DIR, str(i)))

model_dict = pickle.load(open(MODEL_FILE, 'rb'))
model = model_dict['model']

# 加载最新模型（如果存在自适应模型）
if os.path.exists(ADAPTIVE_MODEL_FILE):
    try:
        adaptive_model_dict = pickle.load(open(ADAPTIVE_MODEL_FILE, 'rb'))
        model = adaptive_model_dict['model']
        print("Loaded adaptive model")
    except:
        print("Using base model")

cap = cv2.VideoCapture(CAMERA_ID)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec

# Initialize hands module with improved parameters
hands = mp_hands.Hands(
    static_image_mode=False,  # For real-time detection
    max_num_hands=1,
    min_detection_confidence=0.7,  # Increased confidence threshold
    min_tracking_confidence=0.5
)

# Labels for numbers 0-9
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# 自适应学习相关变量
adaptive_samples = []  # 存储需要保存的自适应样本
adaptive_lock = threading.Lock()
last_retrain_time = time.time()
retrain_interval = 300  # 每5分钟尝试重新训练一次
samples_per_retrain = 20  # 每收集20个样本重新训练一次


def process_landmarks(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        x_.append(x)
        y_.append(y)

    # Normalize coordinates
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))

    return data_aux, x_, y_


def save_adaptive_sample(frame, predicted_label, correct_label):
    """保存自适应学习样本"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{predicted_label}_{correct_label}.jpg"
    class_dir = os.path.join(ADAPTIVE_DATA_DIR, str(correct_label))

    # 确保目录存在
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # 保存图像
    img_path = os.path.join(class_dir, filename)
    cv2.imwrite(img_path, frame)

    return img_path


def adaptive_retrain():
    """使用新收集的样本重新训练模型"""
    global model, last_retrain_time

    print("\nStarting adaptive retraining...")

    # 创建临时工作目录
    temp_dir = './temp_adaptive'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # 准备数据
    data = []
    labels = []

    # 复制原始数据
    shutil.copytree('./data', os.path.join(temp_dir, 'data'))

    # 添加自适应数据
    for class_label in os.listdir(ADAPTIVE_DATA_DIR):
        class_dir = os.path.join(ADAPTIVE_DATA_DIR, class_label)
        if not os.path.isdir(class_dir):
            continue

        # 创建目标目录（如果不存在）
        target_dir = os.path.join(temp_dir, 'data', class_label)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 复制自适应样本
        for img_file in os.listdir(class_dir):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(target_dir, img_file)
            shutil.copy2(src_path, dst_path)

    # 创建数据集
    from dataset_creator import create_dataset, save_dataset

    try:
        data, labels = create_dataset(os.path.join(temp_dir, 'data'))
        temp_data_file = os.path.join(temp_dir, 'adaptive_data.pickle')
        save_dataset(data, labels, temp_data_file)

        # 训练模型
        from model_trainer import load_data, train_model, save_model

        X, y = load_data(temp_data_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        new_model, y_pred = train_model(X_train, X_test, y_train, y_test)

        # 保存新模型
        save_model(new_model, ADAPTIVE_MODEL_FILE)

        # 更新当前模型
        with adaptive_lock:
            model = new_model
            print("Adaptive model updated successfully!")

            # 复制为新的基础模型（可选）
            shutil.copy(ADAPTIVE_MODEL_FILE, MODEL_FILE)

        return True

    except Exception as e:
        print(f"Adaptive retraining failed: {e}")
        return False

    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def adaptive_retrain_daemon():
    """后台自适应学习守护进程"""
    global adaptive_samples, last_retrain_time

    while True:
        time.sleep(10)  # 每10秒检查一次

        current_time = time.time()
        sample_count = 0

        # 获取样本数量
        for class_label in os.listdir(ADAPTIVE_DATA_DIR):
            class_dir = os.path.join(ADAPTIVE_DATA_DIR, class_label)
            if os.path.isdir(class_dir):
                sample_count += len(os.listdir(class_dir))

        # 检查是否满足重新训练条件
        if (sample_count >= samples_per_retrain and
                (current_time - last_retrain_time) > retrain_interval):

            print(f"Collected {sample_count} adaptive samples, starting retraining...")
            last_retrain_time = current_time

            # 执行重新训练
            if adaptive_retrain():
                print("Adaptive learning completed successfully!")
            else:
                print("Adaptive learning failed, will try again later.")


# 启动自适应学习守护线程
retrain_thread = threading.Thread(target=adaptive_retrain_daemon, daemon=True)
retrain_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        prediction_text = ""
        predicted_number = ""
        show_correction_ui = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles(color=(0, 255, 0), thickness=2),
                    mp_drawing_styles(color=(255, 0, 0), thickness=2)
                )

                # Process landmarks
                data_aux, x_, y_ = process_landmarks(hand_landmarks)

                # Draw bounding box
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                predicted_number = labels_dict[int(prediction[0])]
                prediction_text = f"Number: {predicted_number}"

                # Draw prediction
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    prediction_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # 显示自适应学习UI
        cv2.putText(
            frame,
            "Press 'c' to correct prediction",
            (10, H - 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Press 'r' to reload model",
            (10, H - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, H - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        # 如果正在纠正，显示数字选择UI
        if show_correction_ui:
            cv2.rectangle(frame, (0, 0), (W, H), (0, 0, 0), 150)
            cv2.putText(
                frame,
                "Select correct number:",
                (W // 2 - 200, H // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            # 显示数字选项
            for i in range(10):
                x_pos = W // 2 - 200 + (i % 5) * 80
                y_pos = H // 2 + 50 + (i // 5) * 80
                cv2.putText(
                    frame,
                    str(i),
                    (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )

        cv2.imshow('ASL Numbers Detection - Adaptive Learning', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # 重新加载模型
        elif key == ord('r'):
            try:
                if os.path.exists(ADAPTIVE_MODEL_FILE):
                    adaptive_model_dict = pickle.load(open(ADAPTIVE_MODEL_FILE, 'rb'))
                    with adaptive_lock:
                        model = adaptive_model_dict['model']
                    print("Adaptive model reloaded successfully!")
                else:
                    model_dict = pickle.load(open(MODEL_FILE, 'rb'))
                    with adaptive_lock:
                        model = model_dict['model']
                    print("Base model reloaded")
            except Exception as e:
                print(f"Error reloading model: {e}")

        # 纠正预测结果
        elif key == ord('c') and predicted_number:
            show_correction_ui = True
            correction_frame = frame.copy()
            cv2.imshow('ASL Numbers Detection - Adaptive Learning', frame)

            # 等待用户输入正确的数字
            while show_correction_ui:
                key_corr = cv2.waitKey(1) & 0xFF

                # 检查是否按下了0-9的数字键
                if 48 <= key_corr <= 57:  # 0-9的ASCII码
                    correct_number = chr(key_corr)
                    print(f"User correction: {predicted_number} -> {correct_number}")

                    # 保存样本
                    img_path = save_adaptive_sample(correction_frame, predicted_number, correct_number)
                    print(f"Saved adaptive sample: {img_path}")

                    # 重置UI状态
                    show_correction_ui = False
                    break

                # 取消纠正
                elif key_corr == ord('q') or key_corr == 27:  # ESC键
                    show_correction_ui = False
                    break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()