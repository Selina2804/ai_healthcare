import os
import random
from flask import Blueprint, request, jsonify
from .emotion_detector import analyze_emotion
from .chatbot import get_chatbot_response
from .utils import read_base64_image
from flask import Blueprint, request, jsonify
import cv2, base64, numpy as np, pickle, mediapipe as mp
main = Blueprint('main', __name__)

@main.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.json['image']
    frame = read_base64_image(data)
    emotion = analyze_emotion(frame)  
    return jsonify({"emotion": emotion})

@main.route("/chat", methods=["POST"])
def chat():
    message = request.json['message']
    reply = get_chatbot_response(message)
    return jsonify({"reply": reply})

@main.route("/predict_fall", methods=["POST"])
def predict_fall():
    try:
        img = read_img_b64(request.json['image'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        if not res.pose_landmarks:
            return jsonify({"fall_risk": 0})  # Không phát hiện người → 0 nguy cơ

        # Lấy đặc trưng như khi huấn luyện: 33 điểm × (x, y, z, visibility)
        SELECTED_LANDMARKS = list(range(33))
        kp = []
        for idx in SELECTED_LANDMARKS:
            lm = res.pose_landmarks.landmark[idx]
            kp.extend([lm.x, lm.y, lm.z, lm.visibility])  # ✅ đủ 132 đặc trưng

        # Dự đoán từ mô hình
        pred_label = fall_clf.predict(np.array([kp]))[0]
  # output: 0–99 (label)
        
        return jsonify({"fall_risk": int(pred_label)})  # Trả về đúng nhãn huấn luyện
    except Exception as e:
        print(f"Lỗi dự đoán té ngã: {str(e)}")
        return jsonify({"fall_risk": 0})  # fallback nếu lỗi

@main.route("/analyze_kidney", methods=["POST"])
def analyze_kidney():
    try:
        # Phân tích hình ảnh từ camera
        img = read_img_b64(request.json['image'])
        
        # Giả lập phân tích hình ảnh (trong thực tế sẽ dùng model AI)
        # - Phát hiện bọng mắt, thâm quầng, mắt lờ đờ
        risk_score = random.uniform(0, 0.8)  # Giả lập tỉ lệ nguy cơ 0-80%
        
        # Tăng nguy cơ nếu có các dấu hiệu
        if random.random() > 0.7:  # 30% chance có dấu hiệu
            risk_score = min(risk_score + 0.2, 0.8)
        
        return jsonify({
            "risk_score": risk_score,
            "message": "Đã phân tích hình ảnh gương mặt"
        })
    except Exception as e:
        print(f"Lỗi phân tích bệnh thận: {str(e)}")
        return jsonify({"risk_score": 0})

@main.route("/get_kidney_questions", methods=["GET"])
def get_kidney_questions():
    questions = [
        {
            "question": "Bạn có thường xuyên đi tiểu nhiều vào ban đêm không?",
            "key": "nocturia",
            "weight": 0.15
        },
        {
            "question": "Bạn có cảm thấy mệt mỏi, uể oải thường xuyên không?",
            "key": "fatigue",
            "weight": 0.1
        },
        {
            "question": "Bạn có bị phù nề (sưng) ở mắt cá chân hoặc bàn chân không?",
            "key": "swelling",
            "weight": 0.15
        },
        {
            "question": "Bạn có bị tăng huyết áp không?",
            "key": "hypertension",
            "weight": 0.2
        },
        {
            "question": "Bạn có bị tiểu đường không?",
            "key": "diabetes",
            "weight": 0.2
        },
        {
            "question": "Gia đình bạn có ai bị bệnh thận không?",
            "key": "family_history",
            "weight": 0.1
        },
        {
            "question": "Bạn có hút thuốc lá không?",
            "key": "smoking",
            "weight": 0.05
        },
        {
            "question": "Bạn có thường xuyên dùng thuốc giảm đau không?",
            "key": "painkillers",
            "weight": 0.05
        }
    ]
    return jsonify(questions)

@main.route("/evaluate_kidney_risk", methods=["POST"])
def evaluate_kidney_risk():
    try:
        data = request.json
        answers = data.get("answers", [])
        initial_risk = float(data.get("initial_risk", 0))
        
        # Tính toán tổng trọng số từ câu trả lời
        total_weight = sum(float(answer.get("weight", 0)) for answer in answers)
        
        # Tính nguy cơ cuối cùng (không vượt quá 1)
        final_risk = min(initial_risk + total_weight, 1.0)
        
        # Tạo kết luận
        if final_risk >= 0.5:
            conclusion = (
                f"⚠️ CẢNH BÁO: Nguy cơ bệnh thận của bạn là {final_risk*100:.0f}% (CAO).\n\n"
                "Dựa trên phân tích của chúng tôi, bạn có nhiều dấu hiệu cảnh báo bệnh thận.\n\n"
                "🔹 Khuyến nghị:\n"
                "- Đi khám bác sĩ chuyên khoa thận sớm\n"
                "- Xét nghiệm máu (creatinine, ure)\n"
                "- Xét nghiệm nước tiểu\n"
                "- Siêu âm thận\n"
                "- Kiểm soát huyết áp và đường huyết"
            )
        else:
            conclusion = (
                f"✅ KẾT QUẢ: Nguy cơ bệnh thận của bạn là {final_risk*100:.0f}% (THẤP).\n\n"
                "Tuy nhiên bạn vẫn nên:\n\n"
                "🔹 Duy trì lối sống lành mạnh:\n"
                "- Uống đủ 2 lít nước/ngày\n"
                "- Hạn chế muối (<5g/ngày)\n"
                "- Kiểm soát huyết áp\n"
                "- Hạn chế thuốc giảm đau\n"
                "- Tập thể dục đều đặn"
            )
        
        return jsonify({
            "final_risk": final_risk,
            "conclusion": conclusion,
            "status": "success"
        })
    
    except Exception as e:
        print(f"Lỗi đánh giá nguy cơ thận: {str(e)}")
        return jsonify({
            "final_risk": 0,
            "conclusion": "Đã xảy ra lỗi trong quá trình đánh giá",
            "status": "error"
        }), 500


# MediaPipe pose + model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
model_path = os.path.join(os.path.dirname(__file__), "pose_model.pkl")
fall_model_path = os.path.join(os.path.dirname(__file__), "fall_model.pkl")
fall_clf = pickle.load(open(fall_model_path, "rb"))
clf = pickle.load(open(model_path, "rb"))
def read_img_b64(b64):
    data = b64.split(",")[1]
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@main.route('/analyze_medical_report', methods=['POST'])
def analyze_medical_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Đọc file ảnh
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Tiền xử lý ảnh
        img = preprocess_image(img)
        
        # Phân tích bằng AI (giả lập)
        # Trong thực tế, bạn sẽ sử dụng model đã train để phân tích
        analysis_result = analyze_kidney_report(img)
        
        return jsonify({
            'status': 'success',
            'analysis': analysis_result['analysis'],
            'confidence': analysis_result['confidence'],
            'findings': analysis_result['findings']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_image(img):
    """Tiền xử lý ảnh trước khi phân tích"""
    # Resize ảnh
    img = cv2.resize(img, (800, 600))
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    return thresh

def analyze_kidney_report(img):
    """Phân tích báo cáo y tế về bệnh thận"""
    # Đây là phần giả lập - trong thực tế bạn sẽ sử dụng model AI đã train
    
    # Giả lập kết quả phân tích dựa trên dataset bệnh thận
    findings = {
        'age': random.randint(30, 70),
        'bp': random.choice(['Normal', 'High']),
        'sg': round(random.uniform(1.005, 1.025), 3),
        'al': random.choice(['Negative', 'Trace', '1+', '2+', '3+']),
        'su': random.choice(['Negative', 'Trace', '1+', '2+', '3+']),
        'classification': random.choice(['ckd', 'notckd'])
    }
    
    # Tạo báo cáo phân tích
    if findings['classification'] == 'ckd':
        analysis = (
            "⚠️ Kết quả cho thấy dấu hiệu bệnh thận mãn tính (CKD).\n\n"
            "Các chỉ số bất thường:\n"
            f"- Tuổi: {findings['age']}\n"
            f"- Huyết áp: {findings['bp']}\n"
            f"- Tỉ trọng nước tiểu (SG): {findings['sg']}\n"
            f"- Albumin (AL): {findings['al']}\n"
            f"- Đường (SU): {findings['su']}\n\n"
            "Khuyến nghị: Nên đi khám chuyên khoa thận để được chẩn đoán chính xác hơn."
        )
        confidence = random.randint(75, 95)
    else:
        analysis = (
            "✅ Kết quả không phát hiện dấu hiệu bệnh thận mãn tính rõ rệt.\n\n"
            "Các chỉ số chính:\n"
            f"- Tuổi: {findings['age']}\n"
            f"- Huyết áp: {findings['bp']}\n"
            f"- Tỉ trọng nước tiểu (SG): {findings['sg']}\n"
            f"- Albumin (AL): {findings['al']}\n"
            f"- Đường (SU): {findings['su']}\n\n"
            "Lưu ý: Kết quả này không thay thế chẩn đoán của bác sĩ."
        )
        confidence = random.randint(85, 98)
    
    return {
        'analysis': analysis,
        'confidence': confidence,
        'findings': findings
    }


@main.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Đọc file ảnh
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Phân tích hình ảnh (giả lập hoặc sử dụng model thực tế)
        analysis_result = analyze_uploaded_image(img)
        
        return jsonify({
            'status': 'success',
            'analysis': analysis_result['analysis'],
            'confidence': analysis_result['confidence'],
            'findings': analysis_result['findings']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_uploaded_image(img):
    """Phân tích hình ảnh upload"""
    # Phân tích giả lập - trong thực tế sẽ dùng model AI
    # Có thể phát hiện các dấu hiệu bệnh qua hình ảnh
    
    # Phân tích màu da, mắt, các dấu hiệu bất thường
    findings = {
        'skin_tone': random.choice(['Bình thường', 'Nhợt nhạt', 'Vàng da']),
        'eye_condition': random.choice(['Bình thường', 'Thâm quầng', 'Sưng']),
        'face_swelling': random.choice(['Không', 'Nhẹ', 'Rõ rệt']),
        'health_risk': random.choice(['Thấp', 'Trung bình', 'Cao'])
    }
    
    # Tạo báo cáo phân tích
    if findings['health_risk'] == 'Cao':
        analysis = (
            "⚠️ Phát hiện dấu hiệu sức khỏe bất thường:\n\n"
            f"- Màu da: {findings['skin_tone']}\n"
            f"- Tình trạng mắt: {findings['eye_condition']}\n"
            f"- Sưng mặt: {findings['face_swelling']}\n\n"
            "Khuyến nghị: Nên đi khám bác sĩ để kiểm tra sức khỏe tổng quát."
        )
        confidence = random.randint(70, 90)
    else:
        analysis = (
            "✅ Không phát hiện dấu hiệu bất thường rõ rệt:\n\n"
            f"- Màu da: {findings['skin_tone']}\n"
            f"- Tình trạng mắt: {findings['eye_condition']}\n"
            f"- Sưng mặt: {findings['face_swelling']}\n\n"
            "Lưu ý: Kết quả này chỉ mang tính tham khảo."
        )
        confidence = random.randint(80, 95)
    
    return {
        'analysis': analysis,
        'confidence': confidence,
        'findings': findings
    }

@main.route("/check_posture", methods=["POST"])
def check_posture():
    img = read_img_b64(request.json['image'])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)

    if not res.pose_landmarks:
        return jsonify({"correct": True, "result": "Không phát hiện tư thế"})

    # ✅ DÙNG 33 điểm giống khi train
    SELECTED_LANDMARKS = list(range(33))
    kp = []
    for idx in SELECTED_LANDMARKS:
        lm = res.pose_landmarks.landmark[idx]
        kp.extend([lm.x, lm.y, lm.z])  # ⛔️ Không có visibility như trong train

    # ⚠️ Check nếu mô hình huấn luyện KHÔNG có visibility thì cũng KHÔNG nên thêm ở đây
    pred = clf.predict(np.array([kp]))[0]
    return jsonify({
        "correct": bool(pred),
        "result": "Tư thế ĐÚNG" if pred else "Tư thế SAI"
    })


