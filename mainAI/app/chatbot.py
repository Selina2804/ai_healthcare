import google.generativeai as genai
import random
from datetime import datetime

genai.configure(api_key="AIzaSyAyP2dH5eD8yAzm3mPdvQRUjpmV4dMRtoY")



def format_health_response(text):
    """Định dạng câu trả lời y tế với cấu trúc rõ ràng và liên kết tham khảo"""
    # Tạo các phần ngẫu nhiên cho câu trả lời
    greetings = ["Xin chào!", "Chào bạn!", "Cảm ơn bạn đã hỏi!"]
    closings = [
        "Nếu bạn cần thêm thông tin, hãy hỏi tiếp nhé!",
        "Đừng ngần ngại hỏi thêm nếu bạn cần!",
        "Hãy liên hệ bác sĩ nếu triệu chứng nghiêm trọng!"
    ]
    
    # Kiểm tra nếu là chủ đề về thận
    is_kidney_related = any(keyword in text.lower() for keyword in 
                           ["thận", "tiểu đường", "huyết áp", "nước tiểu", "suy thận"])
    
    # Xử lý và định dạng text từ AI
    formatted_lines = []
    in_list = False
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('* '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f'<li>{line[2:]}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            # Xử lý in đậm bằng thẻ strong thay vì **
            formatted_lines.append(line.replace('**', '<strong>', 1).replace('**', '</strong>', 1))
    
    if in_list:
        formatted_lines.append('</ul>')
    
    formatted_text = '\n'.join(formatted_lines)
    
    # Thêm các điểm nhấn quan trọng
    important_points = [
        "Lưu ý quan trọng:",
        "Điều cần nhớ:",
        "Thông tin quan trọng:"
    ]
    
    # Thêm liên kết tham khảo nếu là chủ đề về thận
    kidney_resources = ""
    if is_kidney_related:
        kidney_resources = """
        <br><br>
        <strong>Tài liệu tham khảo về bệnh thận:</strong>
        <ul>
            <li><a href="https://www.kidney.org/atoz/content/about-chronic-kidney-disease" target="_blank">Hiệp hội Thận quốc gia Hoa Kỳ (NKF)</a></li>
            <li><a href="https://www.niddk.nih.gov/health-information/kidney-disease" target="_blank">Viện Tiểu đường, Tiêu hóa và Bệnh thận Quốc gia Hoa Kỳ (NIDDK)</a></li>
            <li><a href="https://www.who.int/news-room/fact-sheets/detail/chronic-kidney-disease" target="_blank">Tổ chức Y tế Thế giới (WHO)</a></li>
            <li><a href="https://vnha.org.vn/" target="_blank">Hội Thận học Việt Nam</a></li>
        </ul>
        """
    
    # Tạo cấu trúc câu trả lời
    response_parts = [
        f'<p class="small p-2 ms-3 mb-1 rounded-3 bg-light">',
        random.choice(greetings),
        '<br><br>',
        formatted_text,
        kidney_resources,  # Chèn liên kết tham khảo nếu có
        '<br><br>',
        f"<strong>{random.choice(important_points)}</strong> Tôi không phải là bác sĩ. Nếu bạn có vấn đề sức khỏe nghiêm trọng, hãy tham khảo ý kiến chuyên gia y tế.",
        '<br><br>',
        random.choice(closings),
        '<br><br>',
        f'<small><i>Tin nhắn được tạo lúc: {datetime.now().strftime("%H:%M:%S")}</i></small>',
        '</p>'
    ]
    
    return ''.join(response_parts)

def get_kidney_questions():
    return [
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

def evaluate_kidney_risk(answers, initial_risk):
    total_weight = 0
    for answer in answers:
        if answer["answer"]:
            total_weight += answer["weight"]
    
    final_risk = min(initial_risk + total_weight, 1.0)  # Tối đa 100%
    return final_risk

def get_kidney_conclusion(final_risk):
 
 
    if final_risk >= 0.5:
        return (
            f"⚠️ Nguy cơ bệnh thận của bạn là {final_risk*100:.0f}% (CAO). "
            "Bạn nên đi khám bác sĩ chuyên khoa thận để được kiểm tra kỹ hơn. "
            "Các xét nghiệm cần làm: xét nghiệm máu, nước tiểu, siêu âm thận."
        )
    else:
        return (
            f"✅ Nguy cơ bệnh thận của bạn là {final_risk*100:.0f}% (THẤP). "
            "Tuy nhiên bạn vẫn nên duy trì lối sống lành mạnh: uống đủ nước, "
            "hạn chế muối, kiểm soát huyết áp và đường huyết."
        )

def get_chatbot_response(message):
    
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        
        # Tạo prompt chi tiết hơn để kiểm soát đầu ra
        prompt = f"""
Bạn là trợ lý AI y tế chuyên nghiệp. Hãy trả lời câu hỏi sau với:
1. Ngôn ngữ tiếng Việt tự nhiên
2. Cấu trúc rõ ràng, dễ hiểu
3. Không sử dụng dấu ** để in đậm
4. Ưu tiên liệt kê bằng gạch đầu dòng khi cần
5. Luôn kèm lời khuyên tham khảo bác sĩ

Câu hỏi: {message}

Hãy trả lời ngắn gọn nhưng đầy đủ thông tin:
        """
        
        response = model.generate_content(prompt)
        return format_health_response(response.text)
        
    except Exception as e:
        print("❌ Lỗi gọi Gemini:", e)
        error_responses = [
            "Hiện tôi đang gặp chút trục trặc. Bạn vui lòng thử lại sau nhé!",
            "Xin lỗi, tôi không thể trả lời ngay lúc này. Vui lòng hỏi lại sau!",
            "Hệ thống đang bận. Bạn có thể đặt câu hỏi lại sau ít phút!"
        ]
        return random.choice(error_responses)
    

    