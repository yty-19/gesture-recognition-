# 文件名: real_time_recognition.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# --- 第一步：加载我们训练好的模型和标准化工具 ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'base_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'standard_scaler.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("错误：找不到模型文件或标准化工具。")
    print("请先运行 model_training.py 来生成它们。")
    exit()

print("正在加载模型和标准化工具...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("加载完成！")

# --- 第二步：初始化 MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# --- 第三步：启动摄像头并开始实时识别 ---
cap = cv2.VideoCapture(0)

print("摄像头已启动。请在窗口前展示手势。按 'q' 键退出。")

while cap.isOpened():
    success, frame = cap.read()  # 读取原始帧
    if not success:
        print("忽略空的摄像头帧。")
        continue

    image = cv2.flip(frame, 1)

    # 将图像从 BGR 格式转换为 RGB 格式，因为MediaPipe需要RGB
    # 注意：我们处理的是已经翻转的图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 为了提高性能，可选地将图像标记为不可写，以通过引用传递
    image.flags.writeable = False
    results = hands.process(image_rgb)
    image.flags.writeable = True

    # 在图像上绘制手势注释
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,  # 直接在翻转后的 image 上绘制
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

            features = np.array(keypoints).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            predicted_gesture = prediction[0]

            try:
                probability = model.predict_proba(features_scaled)
                confidence = np.max(probability)
            except AttributeError:
                confidence = 1.0

            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = image.shape
            text_x = int(wrist_landmark.x * w) - 20
            text_y = int(wrist_landmark.y * h) + 40

            display_text = f"Gesture: {predicted_gesture} ({confidence:.2f})"

            cv2.rectangle(image, (text_x - 5, text_y - 25), (text_x + 250, text_y + 10), (0, 0, 0), -1)
            cv2.putText(image, display_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Real-time Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 第四步：释放资源 ---
hands.close()
cap.release()
cv2.destroyAllWindows()
print("程序已退出。")