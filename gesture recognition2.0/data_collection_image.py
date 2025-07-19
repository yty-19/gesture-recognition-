# 文件名: data_collection_image.py

import os
import cv2
import json
from datetime import datetime


# 这个函数和视频版一样，用于创建目录结构
def create_data_structure():
    base_dir = "gesture_dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for i in range(11):
        gesture_dir = os.path.join(base_dir, f"gesture_{i}")
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
    return base_dir


# 采集手势图片
def capture_gesture_images(gesture_label, num_samples=50):
    base_dir = create_data_structure()
    gesture_dir = os.path.join(base_dir, f"gesture_{gesture_label}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"准备采集手势 {gesture_label} 的图片...")
    print("请将手势保持在画面中央，调整好角度后按 's' 拍照，按 'q' 结束当前手势的采集")

    sample_idx = 0
    while sample_idx < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        # 在画面上显示提示文字
        cv2.putText(frame, f"Gesture {gesture_label} - Sample {sample_idx + 1}/{num_samples}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to take a photo, 'q' to quit",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Image Collection", frame)

        key = cv2.waitKey(1)

        # 如果按下 's' 键，就拍照
        if key == ord('s'):
            image_path = os.path.join(gesture_dir, f"gesture_{gesture_label}_img_sample_{sample_idx + 1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"已保存样本: {image_path}")
            sample_idx += 1

        # 如果按下 'q' 键，就退出
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 主采集程序
if __name__ == "__main__":
    for gesture in range(11):
        capture_gesture_images(gesture, num_samples=50)