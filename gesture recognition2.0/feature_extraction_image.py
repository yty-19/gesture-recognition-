# 文件名: feature_extraction_image.py

import mediapipe as mp
import cv2
import numpy as np
import os
from tqdm import tqdm

# MediaPipe 初始化，注意 static_image_mode 改为 True
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # 专门为静态图片优化
    max_num_hands=1,
    min_detection_confidence=0.5)

# 从单张图片文件中提取手部关键点
def extract_keypoints_from_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            return np.array(keypoints) # 返回 (21, 3) 的数组
    return None

# 批量处理图片数据集
def process_dataset(dataset_path):
    features = []
    labels = []

    print("开始从图片文件提取特征...")

    # --- 修改开始 ---
    # 不再写死 range(11)，而是自动查找所有名为 gesture_* 的目录
    gesture_dirs = [d for d in os.listdir(dataset_path) if
                    os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('gesture_')]

    for gesture_dir_name in sorted(gesture_dirs):  # sorted确保顺序是 gesture_0, gesture_1 ...
        try:
            # 从文件夹名字 "gesture_0" 中提取出数字 0
            gesture_label = int(gesture_dir_name.split('_')[1])
            gesture_dir = os.path.join(dataset_path, gesture_dir_name)
        except (IndexError, ValueError):
            # 如果文件夹名字不规范，就跳过
            continue
        # --- 修改结束 ---

        # 只查找图片文件
        image_files = [f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        if not image_files:  # 如果文件夹是空的，就跳过
            continue

        print(f"处理手势 {gesture_label}...")
        for image_file in tqdm(image_files):
            image_path = os.path.join(gesture_dir, image_file)
            keypoints = extract_keypoints_from_image(image_path)

            if keypoints is not None:
                features.append(keypoints)
                labels.append(gesture_label)

    # 保存特征和标签
    # 注意：这里会覆盖旧的 features.npy 和 labels.npy 文件
    np.save(os.path.join(dataset_path, "features.npy"), features)
    np.save(os.path.join(dataset_path, "labels.npy"), labels)

    print("图片数据集特征提取完成!")
    return features, labels

# 主处理程序
if __name__ == "__main__":
    dataset_path = "gesture_dataset"
    features, labels = process_dataset(dataset_path)