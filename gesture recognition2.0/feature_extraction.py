import mediapipe as mp
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# 提取关键点函数
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理帧
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取21个关键点 (x, y, z)
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
                keypoints_sequence.append(np.array(keypoints))

    cap.release()
    return np.array(keypoints_sequence)


# 数据标准化函数 (Min-Max归一化)
def min_max_normalize(sequence):
    # 转换为(帧数, 21*3)的形状
    original_shape = sequence.shape
    sequence = sequence.reshape(original_shape[0], -1)

    # 计算每列(特征)的最小值和最大值
    min_vals = np.min(sequence, axis=0)
    max_vals = np.max(sequence, axis=0)

    # 避免除以零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    # 归一化
    normalized = (sequence - min_vals) / range_vals

    # 恢复原始形状
    return normalized.reshape(original_shape), min_vals, max_vals


# Z-Score标准化
def z_score_normalize(sequence):
    # 转换为(帧数, 21*3)的形状
    original_shape = sequence.shape
    sequence = sequence.reshape(original_shape[0], -1)

    # 计算每列的均值和标准差
    mean_vals = np.mean(sequence, axis=0)
    std_vals = np.std(sequence, axis=0)

    # 避免除以零
    std_vals[std_vals == 0] = 1

    # 标准化
    normalized = (sequence - mean_vals) / std_vals

    # 恢复原始形状
    return normalized.reshape(original_shape), mean_vals, std_vals


# 批量处理数据集
def process_dataset(dataset_path):
    features = []
    labels = []
    normalization_params = {}

    # 遍历所有手势目录
    for gesture_label in range(11):
        gesture_dir = os.path.join(dataset_path, f"gesture_{gesture_label}")
        video_files = [f for f in os.listdir(gesture_dir) if f.endswith('.avi')]

        print(f"处理手势 {gesture_label}...")
        for video_file in tqdm(video_files):
            video_path = os.path.join(gesture_dir, video_file)

            # 提取关键点
            keypoints = extract_keypoints(video_path)

            if len(keypoints) == 0:
                continue

            # 数据标准化 (这里使用Min-Max归一化)
            normalized, min_vals, max_vals = min_max_normalize(keypoints)

            # 保存归一化参数
            normalization_params[video_file] = {
                "min": min_vals.tolist(),
                "max": max_vals.tolist()
            }

            # 添加到数据集
            features.append(normalized)
            labels.append(gesture_label)

    # 保存特征和标签
    np.save(os.path.join(dataset_path, "features.npy"), features)
    np.save(os.path.join(dataset_path, "labels.npy"), labels)

    # 保存归一化参数
    with open(os.path.join(dataset_path, "normalization_params.json"), "w") as f:
        json.dump(normalization_params, f, indent=4)

    print("数据集处理完成!")
    return features, labels


# 主处理程序
if __name__ == "__main__":
    dataset_path = "gesture_dataset"
    features, labels = process_dataset(dataset_path)