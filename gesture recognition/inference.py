import cv2
import mediapipe as mp
import joblib
import numpy as np

# 1. 加载模型和标准化器
model = joblib.load('gesture_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

# 2. MediaPipe初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5)

# 3. 实时识别循环
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 镜像处理 + 色彩转换
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 关键点检测
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # 提取并预处理关键点
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        palm_base = landmarks[0]
        normalized = [[lm[0] - palm_base[0], lm[1] - palm_base[1]] for lm in landmarks]
        features = scaler.transform([np.array(normalized).flatten()])

        # 模型预测
        gesture_id = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = proba.max()

        # 显示结果 (当置信度>0.8时显示)
        if confidence > 0.8:
            gesture_name = GESTURE_NAMES[gesture_id]
            cv2.putText(frame, f"{gesture_name} ({confidence:.2f})",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()