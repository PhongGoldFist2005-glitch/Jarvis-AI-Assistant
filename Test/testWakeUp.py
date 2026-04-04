import sounddevice as sd
import numpy as np
from openwakeword.model import Model
import openwakeword

openwakeword.utils.download_models()
model = Model(wakeword_models= [r"P:\Program Files\Python313\AI_assistance\Model\models\hey_jarvis_v0.1.onnx"])
samplerate = 16000
chunk = 1280

def callback(indata, frames, time, status):
    audio = (indata.flatten() * 32767).astype("int16")

    prediction = model.predict(audio)

    for word, score in prediction.items():
        print(score)
        if score > 0.5:
            print("Wake word detected:", word)

with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        blocksize=chunk,
        callback=callback):

    print("Listening...")
    sd.sleep(60000)