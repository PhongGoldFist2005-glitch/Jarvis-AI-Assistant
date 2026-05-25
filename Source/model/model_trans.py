from Source.model.model_abs import modelAbstract
import os
from transformers import pipeline

# Model này có chức năng chuyển đổi voice thành text
class modelTranscript(modelAbstract):
    def __init__(self, modelPath):
        super().__init__()
        self._modelPath = modelPath
        self._modelName = os.path.basename(modelPath)
        self._model = pipeline(
            "automatic-speech-recognition",
            model=self._modelPath,
            device=0
        )

        # Avoid duplicate/deprecated suppress-token processor wiring at generate() call time.
        self._model.model.generation_config.suppress_tokens = None
        self._model.model.generation_config.begin_suppress_tokens = None
    
    def useModel(self, audioPath):
        data = self._model(audioPath, generate_kwargs={
                "language": "vi",
                "task": "transcribe"
            })
        return data["text"]


if __name__ == "__main__":
    a = modelTranscript(r"P:\Program Files\Python313\AI_assistance\Model\PhoWhisper-tiny")
    print(a.getModelName())