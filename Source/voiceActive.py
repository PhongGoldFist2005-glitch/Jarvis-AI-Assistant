import asyncio
import os
import threading
from queue import Queue
from pydub import AudioSegment
from pydub.playback import play
import edge_tts
import time

class playSound:
    def __init__(self, outputPath):
        self.outputPath = outputPath
        self.defaultMessage = "Xin chào, tôi là trợ lý ảo của bạn, tôi có thể giúp gì cho bạn"
        self.voice = "vi-VN-HoaiMyNeural"
        
        self.sound_queue = Queue()
        self.is_playing = False
        self.stop_flag = False
        self.worker_thread = threading.Thread(target=self._sound_worker, daemon=True)
        self.worker_thread.start()

    async def _generate_tts(self, message, output_file):
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text=message, voice=self.voice)
                await asyncio.wait_for(communicate.save(output_file), timeout=15.0)
                return True
            except Exception as e:
                print(f"Thử lần {attempt + 1} thất bại: {e}")
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return False

    def _sound_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        output_file = None
        try:
            while True:
                message = self.sound_queue.get()
                if message is None:
                    break

                self.stop_flag = False
                if not str(message).strip():
                    message = self.defaultMessage

                try:
                    self.is_playing = True
                    output_file = f"{self.outputPath}_{int(time.time()*1000)}.mp3"

                    success = loop.run_until_complete(
                        self._generate_tts(message, output_file)
                    )

                    if not success or self.stop_flag:
                        continue

                    audio = AudioSegment.from_file(output_file, format="mp3")
                    play(audio)
                    os.remove(output_file)

                except Exception as e:
                    print(f"TTS Error: {e}")
                finally:
                    self.is_playing = False
                    if output_file and os.path.exists(output_file):
                        os.remove(output_file)
        finally:
            loop.close()

    def requestSound(self, message):
        self.sound_queue.put(message)

    def stop(self):
        self.stop_flag = True
        while not self.sound_queue.empty():
            try:
                self.sound_queue.get_nowait()
            except:
                break
        print("Audio queue đã bị xóa")