#!/usr/bin/env python3
"""
Real AASIST Pipeline Runner
Deploys actual PyTorch AASIST models with MLflow and KServe
"""
import kfp
import os
import sys

def main():
    """Run the AASIST real serving pipeline"""
    
    print("🚀 Starting AASIST Real Serving Pipeline")
    
    # Import the pipeline function
    from kubeflow_pipeline_real import aasist_real_serving_pipeline
    
    # Environment variables (for debugging)
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow_s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    print(f"📋 Environment Check:")
    print(f"  • MLflow Tracking URI: {mlflow_tracking_uri}")
    print(f"  • MLflow S3 Endpoint: {mlflow_s3_endpoint_url}")
    print(f"  • AWS Access Key ID: {aws_access_key_id}")
    print(f"  • AWS Secret Key: {'***' if aws_secret_access_key else 'Not set'}")
    
    # Create KFP client (simple setup like reference)
    try:
        client = kfp.Client()
        print("✅ Connected to Kubeflow Pipelines successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to Kubeflow Pipelines: {e}")
        print("Please check that Kubeflow Pipelines is running and accessible")
        sys.exit(1)
    
    # Compile pipeline (optional - for inspection)
    try:
        print("🔧 Compiling pipeline...")
        kfp.compiler.Compiler().compile(
            aasist_real_serving_pipeline,
            'aasist_real_serving_pipeline.yaml'
        )
        print("✅ Pipeline compiled successfully!")
    except Exception as e:
        print(f"❌ Failed to compile pipeline: {e}")
        sys.exit(1)
    
    # Run pipeline directly from function (like reference)
    try:
        print("🚀 Starting AASIST pipeline execution...")
        print("Pipeline stages:")
        print("  1. Training: Download AASIST files, load model, upload to MLflow")
        print("  2. Deploy: Deploy model with KServe using MLflow predictor")
        
        run = client.create_run_from_pipeline_func(
            aasist_real_serving_pipeline,
            arguments={},  # No arguments needed
            enable_caching=False
        )
        
        print(f"✅ Pipeline submitted successfully!")
        print(f"📊 Run ID: {run.run_id}")
        if hasattr(run, 'run_url'):
            print(f"🔗 View in dashboard: {run.run_url}")
        
        print(f"\n📋 What this pipeline does:")
        print(f"  1️⃣  Downloads AASIST config and model files from GitHub")
        print(f"  2️⃣  Loads AASIST model architecture dynamically")
        print(f"  3️⃣  Packages model with all dependencies for MLflow")
        print(f"  4️⃣  Uploads to MLflow Model Registry as 'aasist-real-model'")
        print(f"  5️⃣  Deploys using KServe MLflow predictor")
        print(f"  6️⃣  Creates InferenceService 'aasist-real' in admin namespace")
        
        print(f"\n📋 Expected Results:")
        print(f"  • MLflow experiment: 'aasist-real-serving'")
        print(f"  • Model registry: 'aasist-real-model'")
        print(f"  • KServe service: 'aasist-real' in admin namespace")
        print(f"  • Inference endpoint with MLflow backend")
        
        print(f"\n🔧 After completion, check:")
        print(f"  kubectl get inferenceservices -n admin | grep aasist-real")
        print(f"  kubectl get pods -n admin | grep aasist-real")
        
        print(f"\n🎉 AASIST pipeline execution initiated!")
        
    except Exception as e:
        print(f"❌ Failed to submit pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 