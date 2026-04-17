import sounddevice as sd
import numpy as np
import time as timelib
from scipy.io.wavfile import write
from modelTrans import modelTranscript
from modelAssist import modelAssistant
from wakeUp import wakeUpModel
from queryEmbed import querryEmbedWord
import json
from voiceActive import playSound
import threading
import queue

class audioPlay:
    def __init__(self, samplerate, channels, wakeUpModelPath, transcriptPath, outputPath, prompt, databasePath, inputJSON, voiceSavePath):
        # Thông tin đã wake up hay chưa.
        self.__play = False
        # Thông tin audio.
        self.__samplerate = samplerate
        self.__channels = channels
        self.__buffer = []
        
        # Ngưỡng quyết định xem người dùng có đang im lặng hay không
        # Tầm -21 là im lặng rồi
        self.__threshold = -20
        # Biến để bắt đầu check sự kiện im lặng
        # Restart ngay mỗi khi bị không coi là im lặng.
        self.__silenceStart = None
        # Số frame thu được cho toàn bộ đoạn ghi âm sau wake word.
        self.__recordedFrames = 0
        # Số frame có năng lượng đủ cao (không phải im lặng).
        self.__speechFrames = 0
        # Điều kiện tối thiểu để coi là một câu hỏi hợp lệ.
        self.__minUtteranceSeconds = 1.2
        self.__minSpeechSeconds = 0.35

        # Khởi tạo mô hình wake up
        # Khi mô hình đánh giá được độ chính xác trên 0.5
        # Tiến hành thay đổi trạng thái play
        # Để bắt đầu ghi âm.
        self.__wakeModelPath = wakeUpModelPath
        self.__wakeModelObject = wakeUpModel(self.__wakeModelPath)
        self.__wakeModel = self.__wakeModelObject.getModel()
        self.__accuracy = 0.5
        # Thư mục chưa file ghi âm
        self.__outputPath = outputPath

        # Gọi mô hình transcript
        self.__transcriptPath = transcriptPath
        self.__transcriptObject = modelTranscript(self.__transcriptPath)
        self.__transModel = self.__transcriptObject.getModel()

        # Gọi mô hình Assistant.
        self.__assistObject = modelAssistant()
        self.__prompt = prompt

        # Sử dụng mô hình query embed để sửa lỗi chính tả và tìm kiếm thông tin trong database
        self.__queryEmbedObject = querryEmbedWord(databasePath, inputJSON, self.__assistObject)

        # Khởi tạo mô hình phát âm
        self.__bufferText = ""
        self.__playSoundObject = playSound(voiceSavePath)

        # Tạo 1 queue túc trực để nhận text từ LLM đầu vào và tự động đọc ở 1 thread khác mà không bị ảnh hưởng bởi quá trình ghi âm ở thread chính.
        self.__audio_queue = queue.Queue()
        def audio_worker():
            while True:
                # queue trống sẽ wait chứ thay vì trả về None
                text = self.__audio_queue.get()
                if text is None:
                    break
                self.__playSoundObject.requestSound(text)
        
        self.__audio_thread = threading.Thread(
            target=audio_worker,
            daemon=True
        )
        self.__audio_thread.start()

    def __callback(self, indata, frames, time, status):
        # Khi model wakeup nhận ra key words
        # tiến hành thay đổi trạng thái để tiến hành ghi âm dữ liệu.
        audioValue = (indata.flatten() * 32767).astype("int16")
        prediction = self.__wakeModel.predict(audioValue)

        for word, score in prediction.items():
            if word == self.__wakeModelObject.getModelName():
                if score > self.__accuracy:
                    # Ngắt audio đang phát nếu có
                    if self.__play:
                        self.__playSoundObject.stop()
                    self.setPlay(True)
                    print("Recording...")

        if self.__play == True:
            self.__buffer.append(indata.copy())
            self.__recordedFrames += frames
            energy = np.log(np.mean(indata**2) + 1e-10)
            # Silence
            if energy < self.__threshold:
                if self.__silenceStart is None:
                    self.__silenceStart = timelib.perf_counter()
                else:
                    silenceDuration = timelib.perf_counter() - self.__silenceStart
                    if silenceDuration > 2.5:
                        # Ngắt ghi âm lưu audio
                        self.__play = False
                        utteranceSeconds = self.__recordedFrames / self.__samplerate
                        speechSeconds = self.__speechFrames / self.__samplerate

                        if utteranceSeconds < self.__minUtteranceSeconds or speechSeconds < self.__minSpeechSeconds:
                            print("Audio quá ngắn hoặc chưa đủ thông tin, bỏ qua transcript.")
                            self.__audio_queue.put("Ồ tôi chưa nghe rõ câu hỏi của bạn, bạn có thể đọc lại giúp tôi câu hỏi được hay không")
                            self.__buffer.clear()
                            self.__silenceStart = None
                            self.__recordedFrames = 0
                            self.__speechFrames = 0
                            return

                        print("Ngắt kết nối đã lưu file audio users")
                        audio = np.concatenate(self.__buffer)
                        maxAbs = np.max(np.abs(audio))
                        if maxAbs > 0:
                            audio = audio / maxAbs
                        audio_int16 = np.int16(audio * 32767)
                        write(self.__outputPath, self.__samplerate, audio_int16)
                        
                        # Transcript file
                        try:
                            text = self.__transcriptObject.useModel(self.__outputPath)
                            print(f"Before correction: {text}")
                            # Bắn API và prompt lên LLM
                            text = self.__queryEmbedObject.query(text, 1)
                            print(f"After correction: {text}")
                            message = self.__prompt + f"\n\n Người dùng:{text}"
                            self.__audio_queue.put("Tôi đã rõ tôi sẽ xử lý yêu cầu của bạn ngay bây giờ.")
                            
                            # Module dùng để đọc text từ LLM
                            self.cleanBufferText()
                            for response in self.__assistObject.useStreamModel(message):
                                print(response, end="", flush=True)
                                self.__bufferText += response


                                # Nếu gặp dấu chấm câu hoặc buffer đã đủ dài thì tiến hành đọc
                                if response in ["\n", "."] and len(self.__bufferText) > 20:
                                    self.__audio_queue.put(str(self.__bufferText))
                                    self.cleanBufferText()
                            
                            # Đọc nốt phần còn lại.
                            if self.__bufferText:
                                self.__audio_queue.put(str(self.__bufferText))
                                self.cleanBufferText()
                            
                        except Exception as e:
                            print("Transcript model error")
                            print(e)
                        finally:
                            self.__buffer.clear()
                            self.__silenceStart = None
                            self.__recordedFrames = 0
                            self.__speechFrames = 0

            # Chỉ 1 chunk không im lặng thì lập tức không tích lũy thời gian im lặng nữa
            else:
                self.__silenceStart = None
                self.__speechFrames += frames
        # Khi ghi âm cảm giác kết thúc tiến hành kết thúc tiến hành
        # chuyển trạng thái và refresh buffer.
        # ! Có 1 vấn đề làm buffer chỉ clear khi gọi hàm callback
        else:
            self.__buffer.clear()
    

    def playRecored(self):
        with sd.InputStream(
            samplerate= self.__samplerate,
            channels= self.__channels,
            dtype= 'float32',
            callback= self.__callback):
            
            # Vòng lặp liên lục record
            print("Starting record !")
            while(1):
                sd.sleep(1000)
    
    def setPlay(self, play):
        self.__play = play
        if play:
            self.__silenceStart = None
            self.__recordedFrames = 0
            self.__speechFrames = 0
        else:
            self.__buffer.clear()
            self.__recordedFrames = 0
            self.__speechFrames = 0
    
    def cleanBufferText(self):
        self.__bufferText = ""
        return self.__bufferText
    
    def getPlay(self):
        return self.__play
    
if __name__ == "__main__":
    samplerate = 16000
    duration = 5
    wakeUpModelPath= r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\models\hey_jarvis_v0.1.onnx"
    outputPath = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Audio\output.wav"
    transcriptPath = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\PhoWhisper-tiny"
    jsonPath = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Database\reversed_word.jsonl"
    voiceSavePath = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Audio\voice.mp3"

    prompt = """
        Bạn là Jarvis, một trợ lý AI thông minh, chính xác và đáng tin cậy được tạo ra để hỗ trợ người dùng trong nhiều lĩnh vực như công nghệ, lập trình, học tập và kiến thức chung.

        Vai trò của bạn:
        - Bạn là một trợ lý cá nhân thông minh giống như Jarvis trong Iron Man: bình tĩnh, logic và chuyên nghiệp.
        - Hỗ trợ người dùng phân tích vấn đề, giải thích khái niệm và đưa ra giải pháp rõ ràng.
        - Có thể hỗ trợ về lập trình (Python, C++, AI/ML, Linux), khoa học dữ liệu và các câu hỏi học thuật.

        Nguyên tắc trả lời:
        1. Luôn trả lời bằng tiếng Việt rõ ràng và tự nhiên trừ khi người dùng yêu cầu ngôn ngữ khác.
        2. Nếu câu hỏi mang tính kỹ thuật:
        - Giải thích ngắn gọn trước.
        - Sau đó đưa ví dụ minh họa hoặc code nếu cần.
        3. Nếu câu hỏi mơ hồ hoặc thiếu thông tin, hãy hỏi lại để làm rõ.
        4. Tránh trả lời lan man, ưu tiên câu trả lời logic, có cấu trúc và dễ hiểu.
        5. Khi giải thích khái niệm phức tạp, hãy chia nhỏ thành từng bước.
        6. Nếu không chắc chắn về thông tin, hãy nói rõ thay vì suy đoán.
        7. Trừ khi thông tin câu hỏi mà bạn nhận được theo bạn có chứa ký tự tên riêng bằng tiếng Anh, nhưng bị ghi nhầm thì hãy ngầm hiểu tên riêng đấy và trả lời câu hỏi.
        8. Không thêm các kỹ tự đặc biệt như dấu *, -, _ vào câu trả lời của bạn.
        9. Hãy cố gắng chỉnh sửa lỗi chính tả trong câu hỏi có người dùng nếu bạn phát hiện ra lỗi về ngữ pháp tiếng Việt, trước khi trả lời câu hỏi của người dùng. Nếu bạn không chắc chắn về lỗi chính tả, hãy trả lời câu hỏi của người dùng như bình thường mà không cần chỉnh sửa lỗi chính tả đó.
        10. Cứ 3 câu thì bạn hãy xuống dòng 1 lần.

        Phong cách giao tiếp:
        - Thân thiện nhưng chuyên nghiệp.
        - Trả lời như một trợ lý kỹ thuật thông minh.
        - Có thể sử dụng danh sách hoặc từng bước để giải thích.

        Mục tiêu của bạn:
        - Giúp người dùng hiểu vấn đề nhanh nhất và giải quyết được vấn đề của họ.
        - Trở thành một trợ lý đáng tin cậy trong học tập, lập trình và nghiên cứu.
        """
    dictResult = {}
    with open(jsonPath, "r") as f:
        for line in f:
            data = json.loads(line)
            dictResult.update(data)
    a = audioPlay(samplerate= samplerate, channels= 1, wakeUpModelPath= wakeUpModelPath, transcriptPath= transcriptPath, outputPath= outputPath, prompt= prompt, databasePath= "./Database/chroma_db", inputJSON= dictResult, voiceSavePath= voiceSavePath)
    a.playRecored()