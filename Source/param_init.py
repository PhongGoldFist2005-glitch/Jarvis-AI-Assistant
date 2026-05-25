class initAudioPlayerParams:
    def __init__(self):
        # Model asssist
        self.model_name = "gemini-2.5-flash"

        # queryChunk
        # Khoảng cách xa nhât để vẫn coi là lỗi chính tả.
        self.threshold_check_chunk = 0.7
        self.db_check_chunk = "words"
        self.chunk_database = r"Source\Database\chroma_db"
        self.inputJSON = r"Database\reversed_word.jsonl"

        # Wake up model
        self.wake_up_model_path = r"models\hey_jarvis_v0.1.onnx"
        self.accuracy_wake_up = 0.5

        # voice active
        self.voice_output_path = r"Audio\voice"

        # model transcript
        self.transcriptPath = r"PhoWhisper-tiny"

        # audio info
        # Thông tin audio.
        self.samplerate = 16000
        self.channels = 1
        # Ngưỡng quyết định xem người dùng có đang im lặng hay không
        # Tầm -21 là im lặng rồi
        self.threshold = -20
        # Điều kiện tối thiểu để coi là một câu hỏi hợp lệ.
        self.minUtteranceSeconds = 1.2
        self.minSpeechSeconds = 0.35

    def outputFormat(self, parser):
        format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        return [
            ("system", """
            Bạn là "Giã Vịt" — một trợ lý AI thông minh, chính xác, chuyên nghiệp.

            Nhiệm vụ:
            - Trả lời câu hỏi người dùng → output_str
            - Tạo bản tóm tắt → output_summary

            QUY TẮC QUAN TRỌNG:
            - Nếu output_str NGẮN (<= 20 từ) → output_summary = output_str
            - Nếu output_str DÀI → tóm tắt ngắn gọn, giữ ý chính

            Phong cách:
            - Ngắn gọn, rõ ràng, không lan man

            RÀNG BUỘC:
            - CHỈ trả về JSON
            - KHÔNG thêm text ngoài JSON

            Ngữ cảnh (nếu có):
            {context}

            Format trả về:
            """ + format_instructions),
                ("human", "Yêu cầu của người dùng: {question}")
        ]
    
    def checkPrompFunction(self, parser):
        format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        return [
            ("system", """
            Bạn là "Giã Vịt" — một trợ lý AI thông minh, chính xác, và chuyên nghiệp.

            Nhiệm vụ:
            - Trả lời chính xác, rõ ràng, có cấu trúc
            - Không bịa
            - Nếu không chắc → nói rõ

            Phong cách:
            - Ngắn gọn, thông minh, không lan man

            Yêu cầu bổ sung:
            1. Xác định có cần RAG không:
            - needRAG = true nếu câu hỏi cần dữ liệu ngoài hoặc kiến thức chuyên sâu
            - false nếu có thể trả lời ngay

            2. Phát hiện lỗi chính tả:
            - Nếu có lỗi chính tả về tên riêng→ text_need_fix = cụm tên riêng bị sai(KHÔNG sửa lại)
            - Nếu không → ""

            3. formatResponse:
            - Trả lời chính
            - Rõ ràng, có cấu trúc

            Trả về đúng JSON theo format:
            """ + format_instructions),
                ("human", "{question}")
        ]