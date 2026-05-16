import sounddevice as sd
import numpy as np
import time as timelib
from scipy.io.wavfile import write
from modelTrans import modelTranscript
from modelAssist import modelAssistant
from wakeUp import wakeUpModel
import json
from voiceActive import playSound
import threading
import queue
from paramInit import initAudioPlayerParams

class audioPlay:
    def __init__(self, outputPath):
        # Thông tin đã wake up hay chưa.
        self.__play = False
        self.__buffer = []
        # Biến để bắt đầu check sự kiện im lặng
        # Restart ngay mỗi khi bị không coi là im lặng.
        self.__silenceStart = None
        # Số frame thu được cho toàn bộ đoạn ghi âm sau wake word.
        self.__recordedFrames = 0
        # Số frame có năng lượng đủ cao (không phải im lặng).
        self.__speechFrames = 0
        self.__params = initAudioPlayerParams()

        # Khởi tạo mô hình wake up
        # Khi mô hình đánh giá được độ chính xác trên 0.5
        # Tiến hành thay đổi trạng thái play
        # Để bắt đầu ghi âm.
        try:
            self.__wakeModelObject = wakeUpModel(self.__params.wake_up_model_path)
            self.__wakeModel = self.__wakeModelObject.getModel()
            if self.__wakeModel is None:
                print("⚠️ Wake word model not loaded. Wake word detection disabled.")
                self.__wakeModelEnabled = False
            else:
                self.__wakeModelEnabled = True
        except Exception as e:
            print(f"⚠️ Error loading wake word model: {e}")
            self.__wakeModelEnabled = False
            self.__wakeModel = None
        
        # Thư mục chưa file ghi âm
        self.__outputPath = outputPath

        # Gọi mô hình transcript
        self.__transcriptObject = modelTranscript(self.__params.transcriptPath)
        self.__transModel = self.__transcriptObject.getModel()

        # Gọi mô hình Assistant.
        self.__assistObject = modelAssistant()

        # Khởi tạo mô hình phát âm
        self.__bufferText = ""
        self.__playSoundObject = playSound(self.__params.voice_output_path)

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
        
        # Chỉ predict nếu model được load thành công
        if self.__wakeModelEnabled and self.__wakeModel is not None:
            try:
                prediction = self.__wakeModel.predict(audioValue)

                for word, score in prediction.items():
                    if word == self.__wakeModelObject.getModelName():
                        if score > self.__params.accuracy_wake_up:
                            # Ngắt audio đang phát nếu có
                            if self.__play:
                                self.__playSoundObject.stop()
                            self.setPlay(True)
                            print("Recording...")
            except Exception as e:
                print(f"⚠️ Wake word prediction error: {e}")

        if self.__play == True:
            self.__buffer.append(indata.copy())
            self.__recordedFrames += frames
            energy = np.log(np.mean(indata**2) + 1e-10)
            # Silence
            if energy < self.__params.threshold:
                if self.__silenceStart is None:
                    self.__silenceStart = timelib.perf_counter()
                else:
                    silenceDuration = timelib.perf_counter() - self.__silenceStart
                    if silenceDuration > 2.5:
                        # Ngắt ghi âm lưu audio
                        self.__play = False
                        utteranceSeconds = self.__recordedFrames / self.__params.samplerate
                        speechSeconds = self.__speechFrames / self.__params.samplerate

                        if utteranceSeconds < self.__params.minUtteranceSeconds or speechSeconds < self.__params.minSpeechSeconds:
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
                        write(self.__outputPath, self.__params.samplerate, audio_int16)
                        
                        # Transcript file
                        try:
                            text = self.__transcriptObject.useModel(self.__outputPath)
                            self.__audio_queue.put("Tôi đã rõ tôi sẽ xử lý yêu cầu của bạn ngay bây giờ.")
                            
                            # Module dùng để đọc text từ LLM
                            self.cleanBufferText()
                            for response in self.__assistObject.run(text):
                                print(response, end="", flush=True)
                                self.__bufferText += response

                                # Nếu gặp dấu câu hoặc buffer đã đủ dài thì tiến hành đọc
                                if any(punct in response for punct in ["\n", ".", "!", "?", "。", "！", "？"]) and len(self.__bufferText.strip()) > 20:
                                    self._push_audio_to_queue(self.__bufferText)
                                    self.cleanBufferText()
                            
                            # Đọc nốt phần còn lại (có nội dung thực sự)
                            if self.__bufferText.strip():
                                self._push_audio_to_queue(self.__bufferText)
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
            samplerate= self.__params.samplerate,
            channels= self.__params.channels,
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
    
    def _push_audio_to_queue(self, text):
        """Validate text trước khi push vào TTS queue"""
        # Strip whitespace và kiểm tra không rỗng
        cleaned_text = text.strip()
        if cleaned_text and len(cleaned_text) > 0:
            self.__audio_queue.put(cleaned_text)
        else:
            print(f"⚠️ Skipped empty text: {repr(text)}")
    
    def getPlay(self):
        return self.__play
    
if __name__ == "__main__":
    outputPath = r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Audio\output.wav"
    a = audioPlay(outputPath)
    a.playRecored()