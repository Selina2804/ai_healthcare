import cv2
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import threading
import speech_recognition as sr
import queue
import time

# Recognizer & Queue để nghe song song
recognizer = sr.Recognizer()
voice_queue = queue.Queue()
ENABLE_VOICE = False
def listen_for_voice():
    while True:
        with sr.Microphone() as source:
            try:
                print("🎤 Đang lắng nghe... (nói: đúng / sai / thoát / té / đứng / ngồi)")
                audio = recognizer.listen(source, phrase_time_limit=2)
                text = recognizer.recognize_google(audio, language="vi-VN").lower()
                print("👂 Bạn nói:", text)

                if "đúng" in text:
                    voice_queue.put("d")
                elif "sai" in text:
                    voice_queue.put("s")
                elif "thoát" in text or "thoat" in text:
                    voice_queue.put("a")
                elif "té" in text or "ngã" in text or "nga" in text or "fall" in text:
                    voice_queue.put("f")
                elif "đứng" in text or "dung" in text:
                    voice_queue.put("u")
                elif "ngồi" in text or "ngoi" in text:
                    voice_queue.put("n")
            except Exception as e:
                # Tùy chọn: in lỗi ra nếu cần debug
                # print(f"Lỗi giọng nói: {e}")
                continue

# Khởi động luồng nhận giọng nói
if ENABLE_VOICE:
    listener_thread = threading.Thread(target=listen_for_voice, daemon=True)
    listener_thread.start()

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

SELECTED_LANDMARKS = list(range(33))  # Dùng toàn bộ pose 33 điểm

UPPER_BODY_CONNECTIONS = [
    (0, 11), (0, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24), (23, 24)
]

data, labels = [], []
cap = cv2.VideoCapture(0)
print("➡️ Nhấn 'd'=ĐÚNG, 's'=SAI, 'a'=THOÁT hoặc nói tương ứng")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)

    action = None

    if res.pose_landmarks:
        landmarks = res.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Vẽ các kết nối xương trên cơ thể
        for start, end in UPPER_BODY_CONNECTIONS:
            x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
            x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Vẽ các điểm khớp đã chọn
        for idx in SELECTED_LANDMARKS:
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        # Lưu keypoints
        keypoints = []
        for idx in SELECTED_LANDMARKS:
            lm = landmarks[idx]
            keypoints.extend([lm.x, lm.y, lm.z])

        # Hướng dẫn hiển thị
        cv2.putText(frame, "Nói/nhấn: [d]=Đúng, [s]=Sai, [f]=Té, [u]=Đứng, [n]=Ngồi, [a]=Thoát",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Nhận từ bàn phím
        key = cv2.waitKey(10)
        if key in [ord("d"), ord("s"), ord("a"), ord("f"), ord("u"), ord("n")]:
            action = chr(key)
        elif not voice_queue.empty():
            action = voice_queue.get()

        # Xử lý hành động
        if action == "d":
            data.append(keypoints)
            labels.append(1)
            print("✅ ĐÃ LƯU: ĐÚNG")
        elif action == "s":
            data.append(keypoints)
            labels.append(0)
            print("⚠️ ĐÃ LƯU: SAI")
        elif action == "f":
            data.append(keypoints)
            labels.append(2)
            print("🚨 ĐÃ LƯU: TÉ NGÃ")
        elif action == "u":
            data.append(keypoints)
            labels.append(3)
            print("🧍‍♂️ ĐÃ LƯU: ĐỨNG")
        elif action == "n":
            data.append(keypoints)
            labels.append(4)
            print("🪑 ĐÃ LƯU: NGỒI")
        elif action == "a":
            print("👋 Thoát chương trình huấn luyện.")
            break

    # Hiển thị frame
    cv2.imshow("Train Pose (Voice + Key)", frame)

cap.release()
cv2.destroyAllWindows()

if not data:
    print("❌ Không có dữ liệu được lưu.")
    exit()

df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("pose_data.csv", index=False)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(data, labels)
pickle.dump(clf, open("pose_model.pkl", "wb"))

print("✅ Đã lưu thành công: pose_data.csv & pose_model.pkl")
