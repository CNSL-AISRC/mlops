from kfp import dsl
import os
import glob
from dotenv import load_dotenv

load_dotenv()


@dsl.component(
    base_image='pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime',
    target_image=f'{os.getenv("PRIVATE_DOCKER_REGISTRY")}/aasist-project/serving-model:{os.getenv("SERVING_MODEL_VERSION")}',
    packages_to_install=[
        "kserve==0.13.1", "kubernetes==26.1.0", "tenacity==9.0.0", 'dotenv'
    ],
)
def serving_comp(model_uri: str, isvc_name: str):
    from kubernetes.client import V1ObjectMeta
    from kserve import (
        constants,
        KServeClient,
        V1beta1InferenceService,
        V1beta1InferenceServiceSpec,
        V1beta1PredictorSpec,
        V1beta1TorchServeSpec,
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
                pytorch=V1beta1TorchServeSpec(
                    storage_uri=model_uri
                )
            )
        )
    )
    client = KServeClient()
    client.create(isvc)
    print("Inference Service created")
    @retry(
        wait=wait_exponential(multiplier=2, min=1, max=10),
        stop=stop_after_attempt(30),
        reraise=True,
    )
    def assert_isvc_created(client, isvc_name):
        assert client.is_isvc_ready(isvc_name), f"Failed to create Inference Service {isvc_name}."
    print("Waiting for Inference Service to be ready")
    assert_isvc_created(client, isvc_name)
    isvc_resp = client.get(isvc_name)
    isvc_url = isvc_resp['status']['address']['url']
    print("Inference URL:", isvc_url)
    print("Inference Service is ready")
    return isvc_url