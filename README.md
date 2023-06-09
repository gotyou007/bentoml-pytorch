# Deploy a pretrained/Fine-tuned hateBERT model in AWS Sagemaker with BentoML

`Transformers` is a library that helps download and fine-tune popular pretrained models for common machine learning tasks. `BentoML` provides native support for serving and deploying models trained from Transformers. BentoML requires Transformers version 4 or above


0. Install dependencies:

```bash
pip install -r ./requirements.txt
```

1. Serving a pretrained Model: Using pretrained models from the Hugging Face does not require saving the model first in the BentoML model store. A custom runner can be implemented to download and run pretrained models at runtime. Services are the core components of BentoML, where the serving logic is defined. Create a file service.py

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

runner = bentoml.Runner(PretrainedModelRunnable, name="pretrained_classifier")

svc = bentoml.Service('pretrained_classification_service', runners=[runner])

@svc.api(input=Text(), output=JSON())
async def detectViolence(input_series: str) -> list:
    return await runner.async_run(input_series)
```

2. Fine-Tuned model:

```python
#example code (use yelp_review_full data to fine-tune "bert-base-cased" model
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
#a smaller subset of the dataset to speed up the fine-tuning
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#Load your model with the number of expected labels
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased", num_labels=5)

#Create a Trainer object with your model, training arguments, training and test datasets.
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

trainer.train()
```
**Saving a Fine-tuned Model**: Once the model is fine-tuned, create a Transformers Pipeline with the model and save to the BentoML model store. By design, only Pipelines can be saved with the BentoML Transformers framework APIs. Models, tokenizers, feature extractors, and processors, need to be a part of the pipeline first before they can be saved.

```python
import bentoml
from transformers import pipeline

unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

bentoml.transformers.save_model(name="unmasker", pipeline=unmasker)
```

3. we can define runner directly by `bentoml.transformers.get(”model”).to_runner()`

```python
%%writefile service.py
import bentoml

from bentoml.io import Text, JSON

runner = bentoml.transformers.get("unmasker:latest").to_runner()

svc = bentoml.Service("unmasker_service", runners=[runner])

@svc.api(input=Text(), output=JSON())
async def unmask(input_series: str) -> list:
    return await runner.async_run(input_series)
```
4.We can now run the BentoML server for our new service in development mode (note I am using the service defined in the pretrained model:
```python
bentoml serve service:svc --reload
import requests

#Send prediction request to the service:
requests.post(
   'http://0.0.0.0:3000/detectViolence',
   headers={"content-type": "application/json"},
   data="say something for test"
).text
```

5. Build Bento: Once the service definition is finalized, we can build the model and service into a bento

```
bentoml build
```
```console
Building BentoML service "pretrained_classification_service:2qgvurwfjojwv6wa" from build context "/Users/li/OMSA/FullStackDL/BentoML/huggingface_deployment".
Locking PyPI package versions.
/Users/li/miniconda3/envs/bentoml/lib/python3.9/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")

██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝

Successfully built Bento(tag="pretrained_classification_service:2qgvurwfjojwv6wa").
```

6. Build docker image: A docker image can be automatically generated from a Bento for production deployment, via the bentoml containerize CLI command

```
bentoml containerize pretrained_classification_service:latest
#check it in docker
docker images
#Run the docker image to start the BentoServer:
docker run -it --rm -p 3000:3000 pretrained_classification_service:2qgvurwfjojwv6wa serve --production
```
7. It is now ready for serving in production! For starters, you can now serve it with the bentoml serve CLI command:
```
bentoml serve pretrained_classification_service:latest --production
```
```console
2023-03-17T23:12:19-0600 [INFO] [cli] Environ for worker 0: set CPU thread count to 10
2023-03-17T23:12:19-0600 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "pretrained_classification_service:latest" can be accessed at http://localhost:3000/metrics.
2023-03-17T23:12:19-0600 [INFO] [cli] Starting production HTTP BentoServer from "pretrained_classification_service:latest" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)
```
8. Deploying Bentos in Sagemaker
 **Prerequisites:**
    - Terraform - [Terraform](https://www.terraform.io/) is a tool for building, configuring, and managing infrastructure.
    - AWS CLI - installed and configured with an AWS account with permission to Sagemaker, Lambda and ECR
configure AWS CLI and login docker
```
aws configure
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin account.dkr.ecr.us-east-1.amazonaws.com
#initiate bentoctl
pip install bentoctl #if you haven't install it
bentoctl operator install aws-sagemaker #if you haven't install it
bentoctl init
```
```console
Welcome! You are now in interactive mode.

This mode will help you setup the deployment_config.yaml file required for
deployment. Fill out the appropriate values for the fields.

(deployment config will be saved to: ./deployment_config.yaml)

api_version: v1
name: pretrained_classification
operator:
    name: aws-sagemaker
template: terraform
spec:
    region: us-east-2
    instance_type: ml.t2.medium
    initial_instance_count: 1
    timeout: 60
    enable_data_capture: False
    destination_s3_uri:
    initial_sampling_percentage: 1
filename for deployment_config [deployment_config.yaml]:
deployment config generated to: deployment_config.yaml
✨ generated template files.
  - ./main.tf
  - ./bentoctl.tfvars
  ```
This will run the bentoctl generate command for you and will generate the main.tf terraform file, which specifies the resources to be created and the bentoctl.tfvars file which contains the values for the variables used in the main.tf file.

**Build and push AWS sagemaker compatible docker image to the AWS ECR repository.**
```
bentoctl build -b pytorch_mnist_service:latest -f deployment_config.yaml
```
```console
🚀 Image pushed!
✨ generated template files.
  - ./bentoctl.tfvars
```
**Apply Deployment with Terraform**
Initialize terraform project. This installs the AWS provider and sets up the terraform folders.
```
terraform init
#Apply terraform project to create Sagemaker deployment
terraform apply -var-file=bentoctl.tfvars -auto-approve
```
```consele
Apply complete! Resources: 1 added, 0 changed, 0 destroyed.
Apply complete! Resources: 1 added, 0 changed, 0 destroyed.
ecr_image_tag = "`awsaccount`.dkr.ecr.us-east-2.amazonaws.com/pretrained_classification:sfx3dagmpogmockr"

endpoint = "https://zwq6dqnty2.execute-api.us-east-2.amazonaws.com/"
```
Test deployed endpoint
```
URL=$(terraform output -json | jq -r .endpoint.value)classify
curl -i \
  --header "Content-Type: application/json" \
  --request POST \
  --data 'input' \
  $URL
  ```
Last but not least, do not forget to **Delete deployment Use the bentoctl destroy command to remove the registry and the deployment**
```
bentoctl destroy -f deployment_config.yaml
```
