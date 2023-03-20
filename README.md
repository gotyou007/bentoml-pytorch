# Deploy a pretrained/Fine-tuned hateBERT model in AWS Sagemaker with BentoML

`Transformers` is a library that helps download and fine-tune popular pretrained models for common machine learning tasks. `BentoML` provides native support for serving and deploying models trained from Transformers. BentoML requires Transformers version 4 or above


0. Install dependencies:

```bash
pip install -r ./requirements.txt
```

1. Serving a Fined-tuned Model:

```python
%%writefile service.py
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
async def detectViolence(input_series: str) -> list:
    return await runner.async_run(input_series)
```

2. Run the service:

```bash
bentoml serve service.py:svc
```

3. Send test request

```
curl -X POST -H "content-type: application/json" --data "[[5.9, 3, 5.1, 1.8]]" http://127.0.0.1:3000/classify
```

4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize iris_classifier_lda:latest
```
