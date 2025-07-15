import kfp
import mlflow
import os

from kfp.dsl import Input, Model, component
from kfp.dsl import InputPath, OutputPath, pipeline, component
from kserve import KServeClient
from mlflow.tracking import MlflowClient
from tenacity import retry, stop_after_attempt, wait_exponential

@component(
    base_image="python:3.11",
    packages_to_install=[
        "torch==2.0.1", 
        "mlflow==2.15.1", 
        "boto3==1.34.162",
        "numpy==1.24.3"
    ]
)
def upload_aasist_to_mlflow(
    model_path: str, 
    run_name: str, 
    model_name: str,
    config_name: str = "AASIST"
) -> str:
    """Load AASIST model and upload to MLflow"""
    import os
    import mlflow
    import torch
    import numpy as np
    
    print(f"Loading AASIST model from: {model_path}")
    print(f"Config: {config_name}")
    
    # Mock AASIST model class for MLflow
    class AASISTModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model_path, config_name):
            self.model_path = model_path
            self.config_name = config_name
            
        def load_context(self, context):
            # In real implementation, load actual AASIST model here
            print(f"Loading model from {self.model_path}")
            
        def predict(self, context, model_input):
            # Mock prediction for demo
            import random
            batch_size = len(model_input) if hasattr(model_input, '__len__') else 1
            scores = [random.uniform(0.1, 0.9) for _ in range(batch_size)]
            predictions = ["bonafide" if s > 0.5 else "spoof" for s in scores]
            return {
                "predictions": predictions,
                "scores": scores
            }
    
    # Start MLflow run
    mlflow.set_experiment("aasist-serving")
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("author", "aasist-pipeline")
        mlflow.set_tag("config", config_name)
        
        # Log model parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("config_name", config_name)
        
        # Create model instance
        model = AASISTModelWrapper(model_path, config_name)
        
        # Define conda environment
        conda_env = {
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.11",
                "pip",
                {
                    "pip": [
                        "torch==2.0.1",
                        "numpy==1.24.3",
                        "soundfile==0.12.1",
                        "mlflow==2.15.1"
                    ]
                }
            ]
        }
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            conda_env=conda_env,
            registered_model_name=model_name
        )
        
        model_uri = f"{run.info.artifact_uri}/model"
        print(f"Model uploaded to MLflow: {model_uri}")
        return model_uri

@component(
    base_image="python:3.11",
    packages_to_install=["kserve==0.13.1", "kubernetes==26.1.0", "tenacity==9.0.0"]
)
def deploy_aasist_with_kserve(model_uri: str, isvc_name: str) -> str:
    """Deploy AASIST model using KServe"""
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

    print(f"Deploying AASIST model from: {model_uri}")
    print(f"InferenceService name: {isvc_name}")
    
    # Create InferenceService specification
    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=V1ObjectMeta(
            name=isvc_name,
            namespace="admin",  # Using admin namespace as shown in kubectl output
            annotations={"sidecar.istio.io/inject": "false"},
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                service_account_name="kserve-controller-s3",
                sklearn=V1beta1SKLearnSpec(
                    storage_uri=model_uri,
                    resources={
                        "limits": {"cpu": "2", "memory": "4Gi"},
                        "requests": {"cpu": "1", "memory": "2Gi"}
                    }
                )
            )
        )
    )
    
    # Deploy with KServe
    client = KServeClient()
    
    # Delete existing service if it exists
    try:
        client.delete(isvc_name, namespace="admin")
        print(f"Deleted existing InferenceService: {isvc_name}")
        import time
        time.sleep(10)
    except:
        print("No existing InferenceService to delete")
    
    # Create new InferenceService
    client.create(isvc, namespace="admin")
    print(f"Created InferenceService: {isvc_name}")

    # Wait for InferenceService to be ready
    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=10),
        stop=stop_after_attempt(30),
        reraise=True,
    )
    def assert_isvc_created(client, isvc_name, namespace="admin"):
        isvc_status = client.get(isvc_name, namespace=namespace)
        ready = False
        for condition in isvc_status.get('status', {}).get('conditions', []):
            if condition.get('type') == 'Ready' and condition.get('status') == 'True':
                ready = True
                break
        assert ready, f"InferenceService {isvc_name} is not ready yet"

    assert_isvc_created(client, isvc_name)
    
    # Get service URL
    isvc_resp = client.get(isvc_name, namespace="admin")
    isvc_url = isvc_resp['status']['address']['url']
    print(f"Inference URL: {isvc_url}")
    
    return isvc_url

# Pipeline definition
ISVC_NAME = "aasist-minimal"
MLFLOW_RUN_NAME = "aasist_serving_minimal"
MLFLOW_MODEL_NAME = "aasist-model"

# Get environment variables for MLflow
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow.mlflow.svc.cluster.local:5000')
mlflow_s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio.minio.svc.cluster.local:9000')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')

@pipeline(name='aasist-minimal-serving-pipeline')
def aasist_minimal_serving_pipeline(
    model_path: str = "/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth",
    config_name: str = "AASIST"
):
    """Minimal AASIST serving pipeline with MLflow and KServe"""
    
    # Stage 1: Upload model to MLflow
    upload_task = upload_aasist_to_mlflow(
        model_path=model_path,
        run_name=MLFLOW_RUN_NAME,
        model_name=MLFLOW_MODEL_NAME,
        config_name=config_name
    ).set_env_variable(name='MLFLOW_TRACKING_URI', value=mlflow_tracking_uri)\
     .set_env_variable(name='MLFLOW_S3_ENDPOINT_URL', value=mlflow_s3_endpoint_url)\
     .set_env_variable(name='AWS_ACCESS_KEY_ID', value=aws_access_key_id)\
     .set_env_variable(name='AWS_SECRET_ACCESS_KEY', value=aws_secret_access_key)
    
    # Stage 2: Deploy with KServe
    deploy_task = deploy_aasist_with_kserve(
        model_uri=upload_task.output,
        isvc_name=ISVC_NAME
    ).set_env_variable(name='AWS_ACCESS_KEY_ID', value=aws_access_key_id)\
     .set_env_variable(name='AWS_SECRET_ACCESS_KEY', value=aws_secret_access_key)

if __name__ == "__main__":
    # Compile pipeline
    kfp.compiler.Compiler().compile(
        aasist_minimal_serving_pipeline,
        'aasist_minimal_serving_pipeline.yaml'
    )
    print("✅ Pipeline compiled successfully!") 