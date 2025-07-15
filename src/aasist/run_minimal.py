#!/usr/bin/env python3
"""
Minimal Runner for AASIST MLflow + KServe Pipeline
Following the first_ml_model pattern
"""
import kfp
import os
import sys

# Get environment variables (you can override these)
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow.mlflow.svc.cluster.local:5000')
mlflow_s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio.minio.svc.cluster.local:9000')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')

print("🤖 AASIST Minimal Serving Pipeline Runner")
print("=" * 45)
print(f"MLflow Tracking URI: {mlflow_tracking_uri}")
print(f"MLflow S3 Endpoint: {mlflow_s3_endpoint_url}")
print(f"AWS Access Key ID: {aws_access_key_id}")
print(f"AWS Secret Key: {'*' * len(aws_secret_access_key)}")

# Import the pipeline
from kubeflow_pipeline_minimal import aasist_minimal_serving_pipeline

# Create KFP client [[memory:3314682]]
client = kfp.Client()

# Define pipeline parameters
model_path = '/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth'
config_name = 'AASIST'

print("\n📋 Pipeline Parameters:")
print(f"  • Model Path: {model_path}")
print(f"  • Config: {config_name}")

# Compile the pipeline
print("\n🔧 Compiling pipeline...")
kfp.compiler.Compiler().compile(
    aasist_minimal_serving_pipeline, 
    'aasist_minimal_serving_pipeline.yaml'
)
print("✅ Pipeline compiled successfully!")

# Run the pipeline
print("\n🚀 Starting pipeline run...")
run = client.create_run_from_pipeline_func(
    aasist_minimal_serving_pipeline, 
    arguments={
        'model_path': model_path,
        'config_name': config_name
    }, 
    enable_caching=False
)

print(f"✅ Pipeline started successfully!")
print(f"📊 Run ID: {run.run_id}")
if hasattr(run, 'run_url'):
    print(f"🔗 View in dashboard: {run.run_url}")

print("\n📋 Next Steps:")
print("1. Monitor the pipeline in Kubeflow dashboard")
print("2. Once complete, check MLflow for the registered model")
print("3. KServe will create an InferenceService named 'aasist-minimal'")
print("4. Check the service with: kubectl get inferenceservices -n admin")
print("\n🎉 Done!") 