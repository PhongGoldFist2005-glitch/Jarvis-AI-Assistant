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

        self.default_user_press = "a"