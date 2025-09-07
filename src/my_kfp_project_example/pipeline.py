from kfp import dsl
from kfp import compiler
from kfp import kubernetes as k8s
from components.download_data import download_data
from components.preprocess_data import preprocess_data
from components.train_model_comp.train_model import train_model
from components.evaluate_model import evaluate_model
from components.serving_comp import serving_comp
from dotenv import load_dotenv
from kfp_manager import KFPClientManager
import os
load_dotenv()

@dsl.pipeline(name=os.getenv('PIPELINE_NAME'), description=os.getenv('PIPELINE_DESCRIPTION'))
def my_pipeline():
    # Step 0: Create PVC volume for data sharing
    pvc1 = k8s.CreatePVC(
        #pvc_name=os.getenv('PVC_NAME'),
        pvc_name_suffix=os.getenv('PVC_NAME_SUFFIX'),
        access_modes=['ReadWriteMany'],
        size=os.getenv('PVC_SIZE'),
        storage_class_name=os.getenv('PVC_STORAGE_CLASS_NAME'),
    )
    
    # Step 1: Download data
    step1 = download_data(
        dataset_url=os.getenv("DATASET_URL"),
        pvc_path=os.getenv('PVC_MOUNT_PATH')
    )
    k8s.mount_pvc(step1, pvc_name=pvc1.outputs['name'], mount_path=os.getenv('PVC_MOUNT_PATH'))
    step1.set_caching_options(False) # disable caching
    
    # Step 2: Preprocess data
    step2 = preprocess_data(zip_file_path=step1.output)
    k8s.mount_pvc(step2, pvc_name=pvc1.outputs['name'], mount_path=os.getenv('PVC_MOUNT_PATH'))
    step2.set_caching_options(False) # disable caching
    
    # Step 3: Train model
    step3 = train_model(processed_data=step2.output, config=os.getenv('CONFIG_PATH'))
    step3.set_env_variable(name="MLFLOW_TRACKING_URI", value=os.getenv("MLFLOW_TRACKING_URI"))
    step3.set_env_variable(name="MLFLOW_S3_ENDPOINT_URL", value=os.getenv("MLFLOW_S3_ENDPOINT_URL"))
    step3.set_env_variable(name="AWS_ACCESS_KEY_ID", value=os.getenv("AWS_ACCESS_KEY_ID"))
    step3.set_env_variable(name="AWS_SECRET_ACCESS_KEY", value=os.getenv("AWS_SECRET_ACCESS_KEY"))
    step3.set_env_variable(name="MLFLOW_RUN_NAME", value=os.getenv("MLFLOW_RUN_NAME"))
    step3.set_env_variable(name="MLFLOW_EXPERIMENT_NAME", value=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    step3.set_env_variable(name="MLFLOW_MODEL_NAME", value=os.getenv("MLFLOW_MODEL_NAME"))
    
    k8s.mount_pvc(step3, pvc_name=pvc1.outputs['name'], mount_path=os.getenv('PVC_MOUNT_PATH'))
    step3.set_gpu_limit(1) # set gpu limit to 1
    
    k8s.set_image_pull_secrets(step3, [os.getenv("IMAGE_PULL_SECRET_NAME")])
    step3.set_caching_options(False) # disable caching
    
    # Step 4: Evaluate model
    step4 = evaluate_model(model=step3.output)
    k8s.mount_pvc(step4, pvc_name=pvc1.outputs['name'], mount_path=os.getenv('PVC_MOUNT_PATH'))
    
    # # Step 5: Serving model
    print(f"isvc_name: {os.getenv('ISVC_NAME')}")
    # # import sys
    # # sys.exit()
    step5 = serving_comp(model_uri=step4.output, isvc_name=os.getenv("ISVC_NAME"))
    step5.set_gpu_limit(1) # set gpu limit to 1
    step5.set_env_variable(name="MLFLOW_TRACKING_URI", value=os.getenv("MLFLOW_TRACKING_URI"))
    step5.set_env_variable(name="MLFLOW_S3_ENDPOINT_URL", value=os.getenv("MLFLOW_S3_ENDPOINT_URL"))
    step5.set_env_variable(name="AWS_ACCESS_KEY_ID", value=os.getenv("AWS_ACCESS_KEY_ID"))
    step5.set_env_variable(name="AWS_SECRET_ACCESS_KEY", value=os.getenv("AWS_SECRET_ACCESS_KEY"))
    step5.set_env_variable(name="PRIVATE_DOCKER_REGISTRY", value=os.getenv("PRIVATE_DOCKER_REGISTRY"))
    step5.set_env_variable(name="SERVING_MODEL_VERSION", value=os.getenv("SERVING_MODEL_VERSION"))
    step5.set_env_variable(name="IMAGE_PULL_SECRET_NAME", value=os.getenv("IMAGE_PULL_SECRET_NAME"))
    k8s.set_image_pull_secrets(step5, [os.getenv("IMAGE_PULL_SECRET_NAME")])
    
    delete_pvc1 = k8s.DeletePVC(
        pvc_name=pvc1.outputs['name'],
    ).after(step5)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=my_pipeline,
        package_path=os.getenv("PIPELINE_PACKAGE_PATH"),
    )
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "true").lower() == "true",
        dex_username=os.getenv("DEX_USERNAME"),
        dex_password=os.getenv("DEX_PASSWORD"),
        dex_auth_type=os.getenv("DEX_AUTH_TYPE", "local"),
    )
    kfp_client = kfp_client_manager.create_kfp_client()
    run = kfp_client.create_run_from_pipeline_package(
        os.getenv("PIPELINE_PACKAGE_PATH"),
        arguments={},
        namespace=os.getenv("KFP_NAMESPACE"),
        experiment_name=os.getenv("KFP_EXPERIMENT_NAME"),
        #enable_caching=False
    )
    

    #print(f"Run ID: {run.id}")
    
