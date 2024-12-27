import cv2
import mediapipe as mp

# 初始化MediaPipe Pose模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 打开摄像头
cap = cv2.VideoCapture(0)  # 使用默认摄像头设备
if not cap.isOpened():
    print("错误: 无法打开摄像头")
    exit()

# 跌倒检测逻辑
def detect_fall(landmarks):
    """
    根据膝盖和臀部的关键点位置来判断是否发生跌倒。
    这里简单使用膝盖和臀部的高度差值作为判断依据。
    """
    # 计算膝盖和臀部的平均高度差值
    left_knee_hip_diff = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y - landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
    right_knee_hip_diff = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    threshold = 0.11  # 跌倒检测阈值
    return left_knee_hip_diff < threshold and right_knee_hip_diff < threshold  # 如果差值小于阈值，则认为发生了摔倒

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理图像并获取姿势标记
    results = pose.process(image_rgb)

    # 提取每个节点的坐标并绘制姿势标记
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 跌倒检测
        if detect_fall(results.pose_landmarks.landmark):
            status_text = "Falling!"
            color = (0, 0, 255)  # 红色表示跌倒
        else:
            status_text = "Safe"
            color = (0, 255, 0)  # 绿色表示安全

        # 在图像上绘制文本
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # 显示结果图像
    cv2.imshow('Pose Detection', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()

