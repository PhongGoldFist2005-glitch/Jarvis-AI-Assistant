import asyncio
import io
import os
import wave
import threading
from queue import Queue
from pydub import AudioSegment
from pydub.playback import play
import edge_tts

class playSound:
    """
    Tối ưu hóa: Thay gTTS bằng edge-tts (offline, ~10x nhanh hơn)
    - requestSound() không chặn - chỉ gửi vào queue rồi return
    - Thread riêng xử lý phát audio tuần tự
    - Có cơ chế stop() để ngắt audio đang phát
    """
    def __init__(self, outputPath=r"P:\Program Files\Python313\AI_assistance\GitModel\AI_assistant\Audio\voice.mp3"):
        self.outputPath = outputPath
        self.defaultMessage = "Xin chào, tôi là trợ lý ảo của bạn, tôi có thể giúp gì cho bạn"
        self.voice = "vi-VN-HoaiMyNeural"  # Giọng Việt tự nhiên
        
        # Tạo queue và thread worker riêng để xử lý audio tuần tự
        self.sound_queue = Queue()
        self.is_playing = False
        self.stop_flag = False
        self.worker_thread = threading.Thread(target=self._sound_worker, daemon=True)
        self.worker_thread.start()
    
    def _sound_worker(self):
        """Thread worker: lấy text từ queue, download + phát tuần tự"""
        while True:
            message = self.sound_queue.get()
            if message is None:  # Signal kết thúc
                break

            # Reset flag ngắt
            self.stop_flag = False
            
            if not message:
                message = self.defaultMessage

            
            try:
                self.is_playing = True
                output_file = self.outputPath
                communicate = edge_tts.Communicate(
                    text=message,
                    voice=self.voice
                )
                
                # Lưu + phát
                asyncio.run(communicate.save(output_file))

                # Nếu được gọi stop(), không phát
                if self.stop_flag:
                    print("Audio bị ngắt")
                    self.is_playing = False
                    continue
                
                audio = AudioSegment.from_file(output_file, format="mp3")
                play(audio)
            
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.is_playing = False
    
    def requestSound(self, message):
        """Không chặn - chỉ gửi vào queue rồi return ngay"""
        self.sound_queue.put(message)
    
    def stop(self):
        """Ngắt audio đang phát và xóa queue"""
        self.stop_flag = True
        # Xóa toàn bộ audio chưa xử lý trong queue
        while not self.sound_queue.empty():
            try:
                self.sound_queue.get_nowait()
            except:
                break
        print("Audio queue đã bị xóa")

if __name__ == "__main__":
    a = playSound()
    a.requestSound("Xin chào, tôi là trợ lý ảo của bạn")
