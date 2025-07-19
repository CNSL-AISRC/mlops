#!/usr/bin/env python3
"""
Simple AASIST Demo Pipeline

This script demonstrates a clean pipeline setup with PVC for data sharing.
"""

import argparse
import kfp
from kfp import dsl, compiler
from kfp import kubernetes as k8s
from components.download_dataset import download_dataset
from components.extract_dataset import extract_dataset
from kfp_manager import KFPClientManager
from dotenv import load_dotenv
import os
import json

load_dotenv()

@dsl.pipeline(
    name="aasist-demo-pipeline",
    description="Demo pipeline for AASIST anti-spoofing with PVC data sharing"
)
def aasist_demo_pipeline(
    config: dict = {},  # Will be populated with default AASIST config
    device: str = "cuda"
):
   
    # Create PVC volume for data sharing
    pvc1 = k8s.CreatePVC(
        pvc_name='aasist-pvc',
        access_modes=['ReadWriteMany'],
        size='20Gi',
        storage_class_name='microk8s-hostpath',
    )
    
    # Step 1: Download Dataset
    download_task = download_dataset(
        dataset_url="http://10.5.110.131:8080/test.zip"
    )
    download_task.set_display_name("Download Dataset")
    k8s.mount_pvc(download_task, pvc_name=pvc1.outputs['name'], mount_path='/data')

    # print(f"Downloaded dataset to: {download_task.output}")
    # Step 2: Extract Dataset
    extract_task = extract_dataset(
       dataset_path=download_task.output
    )
    extract_task.set_display_name("Extract Dataset")
    k8s.mount_pvc(extract_task, pvc_name=pvc1.outputs['name'], mount_path='/data')
   #extract_task.after(download_task)
    
    # Del
    # delete_pvc1 = k8s.DeletePVC(
    #     pvc_name=pvc1.outputs['name']
    # ).after(extract_task)
    

def run_demo_pipeline(config: dict):
    """Run the demo pipeline"""
    
    print("🎵 AASIST Demo Pipeline 🎵")
    print("=" * 40)
    
    # Compile the pipeline
    print("Compiling pipeline...")
    compiler.Compiler().compile(aasist_demo_pipeline, 'aasist_demo_pipeline.yaml')
    print("✅ Pipeline compiled: aasist_demo_pipeline.yaml")
    
    # Create KFP client (simplified version from test.py)
    # Create KFP client
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "true").lower() == "true",
        dex_username=os.getenv("DEX_USERNAME"),
        dex_password=os.getenv("DEX_PASSWORD"),
        dex_auth_type=os.getenv("DEX_AUTH_TYPE", "local"),
    )
    
    kfp_client = kfp_client_manager.create_kfp_client()
    
    
    # Run the pipeline with demo parameters
    run = kfp_client.create_run_from_pipeline_package(
        'aasist_demo_pipeline.yaml',
        arguments={
            'config': config,
            'device': 'cuda'  # Change to 'cpu' if no GPU available
        },
        namespace="admin",
        experiment_name="aasist-demo-experiment",
    )
    
    print(f"✅ Demo pipeline run created: {run.run_id}")
    print(f"🔗 View run: {os.getenv('KFP_API_URL')}/runs/details/{run.run_id}")

def main():
    parser = argparse.ArgumentParser(description="Run AASIST Anti-spoofing Pipeline")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    
    run_demo_pipeline(config)
    
if __name__ == "__main__":
    main() 