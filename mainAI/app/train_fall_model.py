import cv2
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Các điểm pose cần
SELECTED_LANDMARKS = list(range(33))  # toàn bộ

# Các kết nối để vẽ khung xương
POSE_CONNECTIONS = [
    (0, 11), (0, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

# Dữ liệu và nhãn
fall_data = []
fall_labels = []

# Bật camera
cap = cv2.VideoCapture(0)
print("🎥 Nhấn [n]=Bình thường | [w]=Đứng không vững | [p]=Chuẩn bị té | [f]=Đã té | [a]=Dừng và huấn luyện")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    h, w, _ = frame.shape
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Vẽ khung xương
        for start, end in POSE_CONNECTIONS:
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Vẽ điểm
        for idx in SELECTED_LANDMARKS:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Trích xuất đặc trưng
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])  # 132 đặc trưng

        # Xử lý phím nhấn
        key = cv2.waitKey(10) & 0xFF
        if key == ord('n'):
            fall_data.append(features)
            fall_labels.append(0)
            print("✅ Đã ghi: Bình thường (0)")
        elif key == ord('w'):
            fall_data.append(features)
            fall_labels.append(1)
            print("⚠️ Đã ghi: Đứng không vững (1)")
        elif key == ord('p'):
            fall_data.append(features)
            fall_labels.append(2)
            print("❗ Đã ghi: Chuẩn bị té (2)")
        elif key == ord('f'):
            fall_data.append(features)
            fall_labels.append(3)
            print("🚨 Đã ghi: ĐÃ TÉ (3)")
        elif key == ord('a'):
            print("🛑 Dừng và huấn luyện...")
            break

    # Hướng dẫn trên khung hình
    cv2.putText(frame, "[n]=Bình thường | [w]=Không vững | [p]=Chuẩn bị té | [f]=Đã té | [a]=Huấn luyện",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow("Fall Detection Training", frame)

cap.release()
cv2.destroyAllWindows()

# Huấn luyện model
if not fall_data:
    print("❌ Không có dữ liệu nào được thu thập!")
    exit()

df = pd.DataFrame(fall_data)
df["label"] = fall_labels

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Lưu model
with open("fall_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Đã huấn luyện và lưu fall_model.pkl thành công!")
