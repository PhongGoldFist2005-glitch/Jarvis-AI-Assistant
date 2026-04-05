from openwakeword.model import Model
import os
from modelAbs import modelAbstract

class wakeUpModel(modelAbstract):
    # Class lưu giữ thông tin của model.
    def __init__(self, modelPath):
        super().__init__()
        self._modelPath = modelPath
        self._model = Model(wakeword_models=[modelPath])
        # Chỉ dùng cho onnx model.
        self._modelName = os.path.basename(modelPath)[:-5]

if __name__ == "__main__":
    a = wakeUpModel(r"P:\Program Files\Python313\AI_assistance\Model\models\hey_jarvis_v0.1.onnx")
    print(a.getModelName())