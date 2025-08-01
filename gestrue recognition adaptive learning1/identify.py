import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm


# Initialize MediaPipe hands
def init_mediapipe():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )


# Process a single image and extract hand landmarks
def process_image(image_path, hands):
    data_aux = []
    x_ = []
    y_ = []

    # Read and process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get hand landmarks
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None

    # Process landmarks
    hand_landmarks = results.multi_hand_landmarks[0]
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)

    # Normalize coordinates
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))

    return data_aux


# 更新后的create_dataset函数
def create_dataset(data_dirs=['./data']):
    hands = init_mediapipe()
    data = []
    labels = []

    # 计算总图像数
    total_images = 0
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for dir_ in os.listdir(data_dir):
                dir_path = os.path.join(data_dir, dir_)
                if os.path.isdir(dir_path):
                    total_images += len(os.listdir(dir_path))

    if total_images == 0:
        print("No images found in data directories")
        return data, labels

    print(f"Processing {total_images} images...")
    with tqdm(total=total_images) as pbar:
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                continue

            for dir_ in sorted(os.listdir(data_dir)):
                dir_path = os.path.join(data_dir, dir_)
                if not os.path.isdir(dir_path):
                    continue

                for img_file in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_file)

                    data_aux = process_image(img_path, hands)
                    if data_aux is not None:
                        data.append(data_aux)
                        labels.append(dir_)

                    pbar.update(1)

    return data, labels


# Save processed dataset to file
def save_dataset(data, labels, output_file='data.pickle'):
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(data)}")
    print(f"Number of classes: {len(set(labels))}")


if __name__ == "__main__":
    try:
        # 添加自适应数据目录
        data_dirs = ['./data', './adaptive_data']
        data, labels = create_dataset(data_dirs)
        save_dataset(data, labels, 'adaptive_data.pickle')  # 保存为单独的文件
    except Exception as e:
        print(f"Error occurred: {e}")