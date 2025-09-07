from kfp import dsl
from kfp import compiler
# Remove kubernetes import as it's not needed for local execution
from components.download_data import download_data
from components.preprocess_data import preprocess_data
from components.train_model_comp.train_model import train_model
from components.evaluate_model import evaluate_model
from components.serving_comp import serving_comp
from dotenv import load_dotenv
from kfp_manager import KFPClientManager
import os
load_dotenv()

from kfp import local

local.init(runner=local.DockerRunner())

@dsl.pipeline(name=os.getenv('PIPELINE_NAME'), description=os.getenv('PIPELINE_DESCRIPTION'))
def my_pipeline():
    # For local execution, we'll use a shared local directory instead of PVC
    # The components should handle data sharing through local filesystem
    
    # Step 1: Download data
    step1 = download_data(
        dataset_url=os.getenv("DATASET_URL"),
        pvc_path=os.getenv('PVC_MOUNT_PATH')  # This will be treated as local path
    )
    step1.set_caching_options(False) # disable caching
    
    # Step 2: Preprocess data
    step2 = preprocess_data(zip_file_path=step1.output)
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
    
    # Remove GPU limit setting as it may not work in local mode
    # step3.set_gpu_limit(1) # set gpu limit to 1
    
    # Remove image pull secrets as they're Kubernetes-specific
    # k8s.set_image_pull_secrets(step3, [os.getenv("IMAGE_PULL_SECRET_NAME")])
    step3.set_caching_options(False) # disable caching
    
    # Step 4: Evaluate model (currently commented out)
    # step4 = evaluate_model(model=step3.output)
    
    # Step 5: Serving model (currently commented out)
    # step5 = serving_comp(model_uri=step4.output, isvc_name=os.getenv("ISVC_NAME"))
    
    # For local execution, we don't need to delete PVC, just return the final step output
    #return step3.output


if __name__ == "__main__":
   
    pipeline_task = my_pipeline()
    print(f"Pipeline completed successfully. Final output: {pipeline_task}")
    

    #print(f"Run ID: {run.id}")
    
