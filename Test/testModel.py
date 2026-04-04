from transformers import pipeline
import torch

# Tạo pipeline nhận dạng giọng nói
asr = pipeline("automatic-speech-recognition", model=r"P:\Program Files\Python313\AI_assistance\Model\PhoWhisper-tiny", device= 0, torch_dtype=torch.float16)

# Chuyển giọng nói thành text
result = asr(r"P:\Program Files\Python313\AI_assistance\Model\Audio\output.wav")
print(result["text"])