import ollama
from modelAbs import modelAbstract

class modelAssistant(modelAbstract):
    def __init__(self):
        super().__init__()
        self._modelName = "qwen2.5"
        self._model = ollama.Client()
        self._modelPath = "qwen2.5"
    
    def useModel(self, message):
        response = self._model.generate(
            model= self._modelName,
            prompt= message,
            options={
                "num_gpu": 1
            }
        )

        return response["response"]
    
    def useStreamModel(self, message):
        stream = self._model.generate(
            model=self._modelName,
            prompt=message,
            stream=True,
            options={
                "num_gpu": 1
            }
        )

        for chunk in stream:
            yield chunk["response"]