from kfp import dsl
import os
from dotenv import load_dotenv

load_dotenv()

@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "kserve==0.13.1", "kubernetes==26.1.0", "tenacity==9.0.0", 'dotenv'
    ],
)
def serving_comp(model_uri: str, isvc_name: str) -> str:
    from kubernetes.client import V1ObjectMeta, V1LocalObjectReference
    from kserve import (
        constants,
        KServeClient,
        V1beta1InferenceService,
        V1beta1InferenceServiceSpec,
        V1beta1PredictorSpec,
        V1beta1TorchServeSpec,
        V1beta1ModelSpec,
    )
    from tenacity import retry, wait_exponential, stop_after_attempt
    import os
    #     # # Create InferenceService with custom predictor
    print(f"isvc_name: {isvc_name}")
    print(f"model_uri: {model_uri}")
    #print(f"os.getenv('PRIVATE_DOCKER_REGISTRY'): {os.getenv('PRIVATE_DOCKER_REGISTRY')}")
    print(f"os.getenv('SERVING_MODEL_VERSION'): {os.getenv('SERVING_MODEL_VERSION')}")
    # print(f"os.getenv('MLFLOW_TRACKING_URI'): {os.getenv('MLFLOW_TRACKING_URI')}")
    # print(f"os.getenv('MLFLOW_S3_ENDPOINT_URL'): {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
    # print(f"os.getenv('AWS_ACCESS_KEY_ID'): {os.getenv('AWS_ACCESS_KEY_ID')}")
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
                # Use the pre-built custom image that contains your ModelWrapper
                containers=[{
                    "name": "kserve-container",
                    "image": f'{os.getenv("PRIVATE_DOCKER_REGISTRY")}/aasist-project/model-server:{os.getenv("SERVING_MODEL_VERSION")}',
                    "env": [
                        {"name": "MODEL_URI", "value": model_uri},
                        {"name": "MODEL_NAME", "value": isvc_name},
                        {"name": "MLFLOW_TRACKING_URI", "value": os.getenv("MLFLOW_TRACKING_URI", "")},
                        {"name": "MLFLOW_S3_ENDPOINT_URL", "value": os.getenv("MLFLOW_S3_ENDPOINT_URL", "")},
                        {"name": "AWS_ACCESS_KEY_ID", "value": os.getenv("AWS_ACCESS_KEY_ID", "")},
                        {"name": "AWS_SECRET_ACCESS_KEY", "value": os.getenv("AWS_SECRET_ACCESS_KEY", "")},
                    ],
                    "resources": {
                        "requests": {
                            "cpu": "4",
                            "memory": "8Gi"
                        },
                        "limits": {
                            "cpu": "4",
                            "memory": "16Gi"
                        },
                        # add gpu request and limit
                        "gpu": {
                            "requests": {
                                "nvidia.com/gpu": "1"
                            },
                            "limits": {
                                "nvidia.com/gpu": "1"
                            }
                        }
                    },
                }],
                image_pull_secrets=[V1LocalObjectReference(name=os.getenv("IMAGE_PULL_SECRET_NAME"))],
                restart_policy="Always"
            )
        )
    )
    client = KServeClient()
    try:
        existing_isvc = client.get(isvc_name)
        if existing_isvc:
            print(f"Deleting existing InferenceService: {isvc_name}")
            client.delete(isvc_name)
            
            # Wait for deletion to complete
            @retry(
                wait=wait_exponential(multiplier=2, min=1, max=10),
                stop=stop_after_attempt(30),
                reraise=True,
            )
            def wait_for_deletion(client, isvc_name):
                try:
                    client.get(isvc_name)
                    raise Exception(f"InferenceService {isvc_name} still exists")
                except Exception as e:
                    if "not found" in str(e).lower():
                        return True
                    raise e
            
            wait_for_deletion(client, isvc_name)
            print(f"Successfully deleted existing InferenceService: {isvc_name}")
    except Exception as e:
        if "not found" not in str(e).lower():
            print(f"Error checking existing InferenceService: {e}")
    
    print(f"Creating InferenceService: {isvc_name}")
    client.create(isvc)
    
    # Wait for the service to be ready with retry logic
    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=10),
        stop=stop_after_attempt(100),
        reraise=True,
    )
    def assert_isvc_created(client, isvc_name):
        assert client.is_isvc_ready(isvc_name), f"Failed to create Inference Service {isvc_name}."
    
    assert_isvc_created(client, isvc_name)
    
    # Get the service URL
    isvc_resp = client.get(isvc_name)
    isvc_url = isvc_resp['status']['address']['url']
    print(f"✅ InferenceService URL: {isvc_url}")
    
    return isvc_url