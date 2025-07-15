import kfp
import mlflow
import os
import requests

from kfp.dsl import Input, Model, component
from kfp.dsl import InputPath, OutputPath, pipeline, component
from kserve import KServeClient
from mlflow.tracking import MlflowClient
from tenacity import retry, stop_after_attempt, wait_exponential

@component(
    base_image="python:3.11",
    packages_to_install=["requests==2.32.3", "pandas==2.2.2"]
)
def download_dataset(url: str, dataset_path: OutputPath('Dataset')) -> None:
    import requests
    import pandas as pd

    response = requests.get(url)
    response.raise_for_status()

    from io import StringIO
    dataset = pd.read_csv(StringIO(response.text), header=0, sep=";")

    dataset.to_csv(dataset_path, index=False)

@component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "pyarrow==15.0.2"]
)
def preprocess_dataset(dataset: InputPath('Dataset'), output_file: OutputPath('Dataset')) -> None:
    import pandas as pd
    
    df = pd.read_csv(dataset, header=0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.to_parquet(output_file)


@component(
    base_image="python:3.11",
    packages_to_install=["pandas==2.2.2", "scikit-learn==1.5.1", "mlflow==2.15.1", "pyarrow==15.0.2", "boto3==1.34.162"]
)
def train_model(dataset: InputPath('Dataset'), run_name: str, model_name: str) -> str:
    import os
    import mlflow
    import pandas as pd
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(dataset)
    
    target_column = "quality"

    train_x, test_x, train_y, test_y = train_test_split(
        df.drop(columns=[target_column]),
        df[target_column], test_size=0.25,
        random_state=42, stratify=df[target_column]
    )

    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("author", "kf-testing")
        lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        lr.fit(train_x, train_y)
        mlflow.sklearn.log_model(lr, "model", registered_model_name=model_name)
        
        model_uri = f"{run.info.artifact_uri}/model"
        print(model_uri)
        return model_uri

@component(
    base_image="python:3.11",
    packages_to_install=["kserve==0.13.1", "kubernetes==26.1.0", "tenacity==9.0.0"]
)
def deploy_model_with_kserve(model_uri: str, isvc_name: str) -> str:
    from kubernetes.client import V1ObjectMeta
    from kserve import (
        constants,
        KServeClient,
        V1beta1InferenceService,
        V1beta1InferenceServiceSpec,
        V1beta1PredictorSpec,
        V1beta1SKLearnSpec,
    )
    from tenacity import retry, wait_exponential, stop_after_attempt

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=V1ObjectMeta(
            name=isvc_name,
            annotations={"sidecar.istio.io/inject": "false"},
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                service_account_name="kserve-controller-s3",
                sklearn=V1beta1SKLearnSpec(
                    storage_uri=model_uri
                )
            )
        )
    )
    
    client = KServeClient()
    client.create(isvc)

    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=10),
        stop=stop_after_attempt(30),
        reraise=True,
    )
    def assert_isvc_created(client, isvc_name):
        assert client.is_isvc_ready(isvc_name), f"Failed to create Inference Service {isvc_name}."

    assert_isvc_created(client, isvc_name)
    isvc_resp = client.get(isvc_name)
    isvc_url = isvc_resp['status']['address']['url']
    print("Inference URL:", isvc_url)
    
    return isvc_url

ISVC_NAME = "wine-regressor8"
MLFLOW_RUN_NAME = "elastic_net_models_vscode"
MLFLOW_MODEL_NAME = "wine-elasticnet"

mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
print(mlflow_tracking_uri)
mlflow_s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
print(mlflow_s3_endpoint_url)
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
print(aws_access_key_id)
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
print(aws_secret_access_key)

import sys
sys.exit(1)

@pipeline(name='download-preprocess-train-deploy-pipeline')
def download_preprocess_train_deploy_pipeline(url: str):
    download_task = download_dataset(url=url)
    
    preprocess_task = preprocess_dataset(
        dataset=download_task.outputs['dataset_path']
    )
    
    train_task = train_model(
        dataset=preprocess_task.outputs['output_file'], run_name=MLFLOW_RUN_NAME, model_name=MLFLOW_MODEL_NAME
    ).set_env_variable(name='MLFLOW_TRACKING_URI', value=mlflow_tracking_uri)\
     .set_env_variable(name='MLFLOW_S3_ENDPOINT_URL', value=mlflow_s3_endpoint_url)\
     .set_env_variable(name='AWS_ACCESS_KEY_ID', value=aws_access_key_id)\
     .set_env_variable(name='AWS_SECRET_ACCESS_KEY', value=aws_secret_access_key)
    
    deploy_task = deploy_model_with_kserve(
        model_uri=train_task.output, isvc_name=ISVC_NAME
    ).set_env_variable(name='AWS_SECRET_ACCESS_KEY', value=aws_secret_access_key)

client = kfp.Client()

url = 'https://raw.githubusercontent.com/canonical/kubeflow-examples/main/e2e-wine-kfp-mlflow/winequality-red.csv'

kfp.compiler.Compiler().compile(download_preprocess_train_deploy_pipeline, 'download_preprocess_train_deploy_pipeline.yaml')

run = client.create_run_from_pipeline_func(download_preprocess_train_deploy_pipeline, arguments={'url': url}, enable_caching=False)