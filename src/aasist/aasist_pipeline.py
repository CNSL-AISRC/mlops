from kfp import dsl, compiler
from kfp_manager import KFPClientManager
from dotenv import load_dotenv
import os
import json

# Import all components
from components.download_dataset import download_dataset
from components.extract_dataset import extract_dataset
# from components.preprocessing_dataset import preprocessing_dataset
from components.init_model import init_model

load_dotenv()
#aasist-anti-spoofing-pipeline

@dsl.pipeline(
    name="test2",
    description="End-to-end pipeline for training and deploying AASIST anti-spoofing models"
)
def aasist_pipeline(
    config_json: str = "",  # Will be populated with default AASIST config
    device: str = "cuda"
):
    """
    Complete AASIST Anti-spoofing Pipeline
    
    This pipeline:
    1. Downloads and prepares the ASVspoof2019 dataset
    2. Preprocesses the data for training
    3. Initializes the AASIST model
    4. Trains the model
    5. Evaluates the model performance
    6. Deploys the model to MLFlow and optionally KServe
    """
    
    # Step 1: Download Dataset using simple string return
    download_task = download_dataset(
        dataset_url="http://10.5.110.131:8080/test.zip",
    )
    download_task.set_display_name("Download Dataset")
    
    # Step 2: Extract Dataset using simple string parameter
    extract_task = extract_dataset(
        zip_file_path=download_task.output
    )
    extract_task.set_display_name("Extract & Validate Dataset")
    extract_task.after(download_task)


    # Step 3: Preprocessing Dataset
    # preprocessing_task = preprocessing_dataset(
    #     dataset_path=extract_task.output,
    #     track=track,
    #     batch_size=batch_size,
    #     seed=seed,
    #     data_ratio=data_ratio,
    #     cut_length=64600
    # )
    # preprocessing_task.set_display_name("Preprocess Dataset")
    
    # preprocessing_task.after(extract_task)
    
    # Step 4: Initialize Model - ALSO COMMENTED
    # init_model_task = init_model(
    #     config_str=config_json,
    #     device=device
    # )
    # init_model_task.set_display_name("Initialize Model")
    
    # Step 5: Training
    # training_task = training(
    #     model_path=init_model_task.output,
    #     preprocessing_info=preprocessing_task.output,
    #     config_str=config_json,
    #     num_epochs=num_epochs,
    #     learning_rate=learning_rate,
    #     weight_decay=weight_decay,
    #     device=device
    # )
    # training_task.set_display_name("Train Model")
    # training_task.after(init_model_task, preprocessing_task)
    
    # Step 6: Evaluation
    # evaluation_task = evaluation(
    #     trained_model_path=training_task.output,
    #     preprocessing_info=preprocessing_task.output,
    #     device=device
    # )
    # evaluation_task.set_display_name("Evaluate Model")
    # evaluation_task.after(training_task)
    
    # Step 7: Serving
    # serving_task = serving(
    #     trained_model_path=training_task.output,
    #     evaluation_results=evaluation_task.output,
    #     mlflow_tracking_uri=mlflow_tracking_uri,
    #     mlflow_experiment_name=mlflow_experiment_name,
    #     model_name=model_name,
    #     model_version=model_version,
    #     deploy_to_kserve=deploy_to_kserve,
    #     kserve_namespace=kserve_namespace
    # )
    # serving_task.set_display_name("Deploy Model")
    # serving_task.after(evaluation_task)

