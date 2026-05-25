from abc import ABC

# Đây là lớp trừu tượng cho các mô hình khác nhau (onnx, torch, tensorflow, ...)
class modelAbstract(ABC):
    def __init__(self):
        super().__init__()
        self._modelPath = None
        self._model = None
        # Chỉ dùng cho onnx model.
        self._modelName = None
    
    def getModel(self):
        return self._model
    def getModelPath(self):
        return self._modelPath
    def getModelName(self):
        return self._modelName