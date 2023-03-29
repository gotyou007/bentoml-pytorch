import bentoml

from bentoml.io import Text, JSON
from transformers import pipeline

class PretrainedModelRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.classifier = pipeline(task="text-classification", model='GroNLP/hateBERT')

    @bentoml.Runnable.method(batchable=False)
    def __call__(self, input_text):
        return self.classifier(input_text)

runner = bentoml.Runner(PretrainedModelRunnable, name="pretrained_unmasker")

svc = bentoml.Service('pretrained_classification_service', runners=[runner])

@svc.api(input=Text(), output=JSON())
async def detectViolence(text: str) -> list:
    return await runner.async_run(text)
