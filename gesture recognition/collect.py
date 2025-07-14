import os
import cv2
import numpy as np
import json
import time
from datetime import datetime


# 创建标准化的采集目录结构
def create_data_structure():
    base_dir = "gesture_dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 创建0-6手势目录
    for i in range(7):
        gesture_dir = os.path.join(base_dir, f"gesture_{i}")
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)

    # 创建元数据文件
    metadata = {
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gestures": ["0", "1", "2", "3", "4", "5", "6"],
        "resolution": "1280x720",
        "fps": 30,
        "background": "uniform_light_gray",
        "lighting": "consistent_indoor_lighting",
        "distance": "arm_length",
        "hand_orientation": "palm_facing_camera"
    }

    with open(os.path.join(base_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return base_dir


# 采集手势视频
def capture_gesture_videos(gesture_label, num_samples=50):
    base_dir = create_data_structure()
    gesture_dir = os.path.join(base_dir, f"gesture_{gesture_label}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"准备采集手势 {gesture_label} 的视频...")
    print("请将手势保持在画面中央，按 's' 开始采集，按 'q' 结束")

    for sample_idx in range(num_samples):
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # 显示采集提示
            cv2.putText(frame, f"Gesture {gesture_label} - Sample {sample_idx + 1}/{num_samples}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to start recording, 'q' to quit",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Gesture Collection", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # 录制5秒视频
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(gesture_dir, f"gesture_{gesture_label}_sample_{sample_idx + 1}.avi")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))

        start_time = time.time()
        while time.time() - start_time < 5:  # 录制5秒
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow("Recording...", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        out.release()
        cv2.destroyWindow("Recording...")
        print(f"已保存样本: {video_path}")

    cap.release()
    cv2.destroyAllWindows()


# 主采集程序
if __name__ == "__main__":
    for gesture in range(7):
        capture_gesture_videos(gesture, num_samples=50)