import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

samplerate = 16000
duration = 5

buffer = []

def callback(indata, frames, time, status):
    buffer.append(indata.copy())

with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        callback=callback):

    print("Nói đi...")
    sd.sleep(duration * 1000)

audio = np.concatenate(buffer)

audio = audio / np.max(np.abs(audio))
audio_int16 = np.int16(audio * 32767)

write("output.wav", samplerate, audio_int16)

print("Saved.")