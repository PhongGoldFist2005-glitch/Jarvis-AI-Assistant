from modelAbs import modelAbstract
import os
from transformers import pipeline

class modelTranscript(modelAbstract):
    def __init__(self, modelPath):
        super().__init__()
        self._modelPath = modelPath
        self._modelName = os.path.basename(modelPath)
        self._model = pipeline(
            "automatic-speech-recognition",
            model=self._modelPath,
            device=0,
            generate_kwargs={
                "language": "vi",
                "forced_decoder_ids": None
            }
        )
    
    def useModel(self, audioPath):
        data = self._model(audioPath)
        return data["text"]


if __name__ == "__main__":
    a = modelTranscript(r"P:\Program Files\Python313\AI_assistance\Model\PhoWhisper-tiny")
    print(a.getModelName())