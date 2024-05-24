import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp
import requests

# 加载训练好的模型
model = load_model('fall_detection_model.h5')

def preprocess_frame_sequence(frame_sequence, max_frames=50):
    # 将帧序列填充/截断到max_frames长度
    padded_sequence = pad_sequences(frame_sequence, maxlen=max_frames, padding='post', truncating='post', dtype='float32')
    # 添加额外的维度
    padded_sequence = np.repeat(np.expand_dims(padded_sequence, axis=-1), 3, axis=-1)
    return padded_sequence

# 配置摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

frame_sequence = []

# 初始化MediaPipe姿态模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像转换为RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #设置 image.flags.writeable = False 可以防止对图像数组的修改
    image.flags.writeable = False

    # 处理图像并检测姿态
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制检测到的骨架
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #mp_pose.POSE_CONNECTIONS：这是一个包含姿态关键点连接关系的常量，定义了关键点之间的连接。

        # 提取骨架点
        skeleton_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
        frame_sequence.append(skeleton_points)

        # 如果序列长度超过max_frames，删除最旧的帧
        if len(frame_sequence) > 50:
            frame_sequence.pop(0)

        # 如果序列长度足够，进行预测
        if len(frame_sequence) == 50:
            # 预处理帧序列
            #原始输入数据是一个形状为 (50, 33, 3)
            input_sequence = preprocess_frame_sequence([frame_sequence])
            # 直接使用 input_sequence 进行预测,通过 np.expand_dims 将其变为 (50, 33, 3, 3)
            prediction = model.predict(input_sequence)
            fall_detected = prediction[0] > 0.5

            # 在帧上显示预测结果
            if fall_detected:
                cv2.putText(image, "Fall Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "No Fall", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示帧
    cv2.imshow('Fall Detection', image)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
