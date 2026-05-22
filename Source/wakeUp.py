from openwakeword.model import Model
import os
from modelAbs import modelAbstract

class wakeUpModel(modelAbstract):
    # Class lưu giữ thông tin của model.
    def __init__(self, modelPath):
        super().__init__()
        self._modelPath = modelPath
        
        try:
            # Tenta carregar từ đường dẫn cục bộ trước
            if os.path.exists(modelPath):
                self._model = Model(wakeword_models=[modelPath])
            else:
                print(f"Model file not found: {modelPath}")
                print("   Trying to load with just filename...")
                # Nếu không tìm thấy file, thử chỉ dùng tên
                model_name = os.path.basename(modelPath).replace('.onnx', '').replace('.tflite', '')
                self._model = Model(wakeword_models=[model_name])
        except Exception as e:
            print(f"⚠️ Error loading wake word model: {e}")
            # Fallback to default
            self._model = None
            
        # Chỉ dùng cho onnx model.
        self._modelName = os.path.basename(modelPath)[:-5]