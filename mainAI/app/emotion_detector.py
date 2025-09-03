from deepface import DeepFace
import cv2
import numpy as np
import base64
import math
import mediapipe as mp

# Bản đồ cảm xúc Anh → Việt
emotion_vi = {
    'angry': 'Tức giận',
    'disgust': 'Ghê tởm',
    'fear': 'Sợ hãi',
    'happy': 'Vui vẻ',
    'sad': 'Buồn bã',
    'surprise': 'Ngạc nhiên',
    'neutral': 'Bình thường'
}

# Khởi tạo cascade nhận diện khuôn mặt và mắt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# MediaPipe face mesh cho phát hiện ngáp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Decode ảnh base64 và resize
def decode_and_resize_image(base64_string):
    try:
        if ',' in base64_string:
            content = base64_string.split(',')[1]
        else:
            content = base64_string

        img_data = base64.b64decode(content)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("⚠️ Ảnh rỗng hoặc lỗi decode.")
            return None

        img = cv2.resize(img, (640, 480))
        return img
    except Exception as e:
        print("❌ Lỗi khi decode ảnh:", e)
        return None

# Phân tích ánh mắt (tỉnh táo, buồn sâu, nhắm mắt...)
def analyze_eye_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            return "Nhắm mắt"
        elif len(eyes) == 1:
            return "1 mắt mở"
        elif len(eyes) >= 2:
            eye_heights = [eh for (_, ey, _, eh) in eyes]
            avg_eye_height = np.mean(eye_heights)
            if avg_eye_height < 10:
                return "Nhắm mắt"
            elif avg_eye_height < 18:
                return "Buồn sâu"
            else:
                return "Tỉnh táo"
    return "Không rõ ánh mắt"

# Khoảng cách euclid giữa 2 điểm
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Nhận diện ngáp bằng landmark miệng
def detect_yawn(frame):
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)

        if not result.multi_face_landmarks:
            return "Không rõ trạng thái ngáp"

        landmarks = result.multi_face_landmarks[0].landmark

        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        mouth_open = euclidean((top_lip.x, top_lip.y), (bottom_lip.x, bottom_lip.y))
        mouth_width = euclidean((left_mouth.x, left_mouth.y), (right_mouth.x, right_mouth.y))

        ratio = mouth_open / mouth_width

        if ratio > 0.35:
            return "Có thể đang ngáp"
        else:
            return "Không buồn ngủ"
    except:
        return "Không rõ trạng thái ngáp"



# Độ ưu tiên cảm xúc (số càng thấp càng ưu tiên)
emotion_priority = {
    'neutral': 1,
    'angry': 3,
    'surprise': 2,
    'sad': 3,
    'fear': 4,
    'happy': 2,
    'disgust': 6
}

# Trọng số ảnh hưởng của độ ưu tiên (nhỏ hơn 1 để giảm ảnh hưởng)
priority_weight = 0.3

def analyze_emotion(frame):
    if frame is None:
        return "Không xác định"

    backends = ['mediapipe', 'opencv']
    result = None

    for backend in backends:
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=backend
            )
            print(f"✅ Sử dụng backend: {backend}")
            break
        except Exception as e:
            print(f"⚠️ Backend {backend} thất bại:", e)
            continue

    if not result or 'emotion' not in result[0]:
        return "Không nhìn thấy khuôn mặt"

    emotions = result[0]['emotion']

    def score(item):
        emotion, percent = item
        priority = emotion_priority.get(emotion, 999)

        # Tách riêng logic sad vs fear
        if emotion == 'fear' and 'sad' in emotions:
            diff = percent - emotions['sad']
            if -3 < diff < 3:  # nếu rất gần nhau
                return 999  # đẩy fear xuống
        return -percent + priority_weight * priority

    prioritized_emotion = min(emotions.items(), key=score)[0]
    eye_status = analyze_eye_movement(frame)
    yawn_status = detect_yawn(frame)

    # Nếu đang ngáp và cảm xúc không mạnh -> Ưu tiên bình thường
    if yawn_status == "Có thể đang ngáp":
        max_emotion = max(emotions.items(), key=lambda x: x[1])
        if max_emotion[1] < 70:  # nếu cảm xúc mạnh nhất < 60%
            prioritized_emotion = 'neutral'

    emotion_vi_name = emotion_vi.get(prioritized_emotion, 'Không xác định')
    return f"{emotion_vi_name} ({eye_status}, {yawn_status})"
