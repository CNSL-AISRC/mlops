#!/usr/bin/env python3
"""
Simple runner script for AASIST MLflow + Serving Pipeline
Handles uploading pretrained models to MLflow and serving them
"""
import os
import sys
from pathlib import Path

def compile_mlflow_pipeline():
    """Compile the MLflow serving pipeline"""
    print("🔧 Compiling AASIST MLflow + Serving Pipeline...")
    
    try:
        from kubeflow_pipeline_mlflow_serving import aasist_mlflow_serving_pipeline
        import kfp
        
        # Compile pipeline
        kfp.compiler.Compiler().compile(
            aasist_mlflow_serving_pipeline, 
            'aasist_mlflow_serving_pipeline.yaml'
        )
        
        print("✅ MLflow serving pipeline compiled successfully!")
        print(f"📄 Generated: aasist_mlflow_serving_pipeline.yaml")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have kubeflow-pipelines installed:")
        print("pip install kfp mlflow")
        return False
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False

def show_mlflow_instructions():
    """Show instructions for using the MLflow serving pipeline"""
    print("\n" + "="*70)
    print("🚀 AASIST MLFLOW + SERVING PIPELINE - USAGE GUIDE")
    print("="*70)
    
    print("\n📋 Three Operation Modes:")
    print("  1️⃣  upload_only - Upload your pretrained model to MLflow")
    print("  2️⃣  serve_only - Load existing MLflow model and serve it")
    print("  3️⃣  upload_and_serve - Upload model then deploy serving")
    
    print("\n🔧 Pipeline Parameters:")
    print("  • model_path: Path/URL to your pretrained model")
    print("  • model_name: Name for MLflow model registry")
    print("  • model_version: Version string (e.g., '1.0', '2.1')")
    print("  • model_stage: MLflow stage (None, Staging, Production)")
    print("  • config_name: Model config (AASIST, AASIST-L)")
    print("  • operation: Mode to run (upload_only/serve_only/upload_and_serve)")
    
    print("\n💡 Usage Examples:")
    
    print("\n  📤 Upload Model Only:")
    print("    operation: 'upload_only'")
    print("    model_path: '/path/to/your/model.pth'")
    print("    model_name: 'my_aasist_model'")
    print("    model_version: '1.0'")
    print("    model_stage: 'Staging'")
    
    print("\n  🚀 Serve Existing Model:")
    print("    operation: 'serve_only'")
    print("    model_name: 'my_aasist_model'")
    print("    model_version: 'latest' (or specific version)")
    print("    model_stage: 'Production'")
    
    print("\n  🔄 Full Workflow:")
    print("    operation: 'upload_and_serve'")
    print("    model_path: 'https://example.com/model.pth'")
    print("    model_name: 'production_aasist'")
    print("    model_version: '2.0'")
    print("    model_stage: 'Production'")
    
    print("\n📊 Supported Model Paths:")
    print("  • Local files: '/path/to/model.pth'")
    print("  • Local directories: '/path/to/model_folder/'")
    print("  • HTTP URLs: 'https://example.com/model.pth'")
    print("  • Cloud storage: 's3://bucket/model.pth', 'gs://bucket/model.pth'")
    
    print("\n🎯 Perfect for:")
    print("  • Production deployment of trained models")
    print("  • Model versioning and A/B testing")
    print("  • Quick serving without retraining")
    print("  • MLflow Model Registry management")
    
    print("\n⚡ Expected Runtime:")
    print("  • upload_only: 2-3 minutes")
    print("  • serve_only: 1-2 minutes")
    print("  • upload_and_serve: 3-5 minutes")
    
    print("\n" + "="*70)

def show_mlflow_setup():
    """Show MLflow setup instructions"""
    print("\n🔧 MLflow Setup (Optional):")
    print("  If you have an MLflow tracking server, set these environment variables:")
    print("  export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000")
    print("  export MLFLOW_S3_ENDPOINT_URL=http://your-s3-endpoint")
    print("  export AWS_ACCESS_KEY_ID=your-access-key")
    print("  export AWS_SECRET_ACCESS_KEY=your-secret-key")
    print("\n  Without these, MLflow will use local file storage.")

def show_example_scenarios():
    """Show example usage scenarios"""
    print("\n📚 Example Scenarios:")
    
    print("\n  🔬 Research Scenario:")
    print("    You trained an AASIST model locally and want to:")
    print("    1. Upload it to MLflow for tracking")
    print("    2. Deploy it for testing")
    print("    → Use: operation='upload_and_serve'")
    
    print("\n  🏭 Production Scenario:")
    print("    You have a trained model and want to:")
    print("    1. Version it in MLflow Registry")
    print("    2. Deploy the 'Production' stage model")
    print("    → Use: operation='upload_only' then operation='serve_only'")
    
    print("\n  🔄 Model Update Scenario:")
    print("    You want to serve a newer version of existing model:")
    print("    1. Upload new version to MLflow")
    print("    2. Transition to Production stage")
    print("    3. Deploy the updated model")
    print("    → Use: operation='upload_and_serve' with new version")
    
    print("\n  🧪 A/B Testing Scenario:")
    print("    You want to compare two model versions:")
    print("    1. Deploy model v1.0 as service-a")
    print("    2. Deploy model v2.0 as service-b")
    print("    3. Compare performance")
    print("    → Use: operation='serve_only' for each version")

def check_environment():
    """Check if environment is set up correctly"""
    print("🔍 Checking environment...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check required files
    required_files = [
        "kubeflow_pipeline_mlflow_serving.py",
        "kubeflow_pipeline_demo.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"Missing file: {file}")
    
    # Check if packages are available
    try:
        import kfp
        print(f"  ✓ Kubeflow Pipelines: {kfp.__version__}")
    except ImportError:
        issues.append("kubeflow-pipelines not installed (pip install kfp)")
    
    try:
        import mlflow
        print(f"  ✓ MLflow: {mlflow.__version__}")
    except ImportError:
        issues.append("mlflow not installed (pip install mlflow)")
    
    # Check MLflow configuration
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_uri:
        print(f"  ✓ MLflow Tracking URI: {mlflow_uri}")
    else:
        print("  ℹ️  MLflow Tracking URI: Using local file storage")
    
    if issues:
        print("❌ Environment issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✅ Environment looks good!")
        return True

def main():
    """Main function"""
    print("🤖 AASIST MLflow + Serving Pipeline Runner")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues before proceeding")
        sys.exit(1)
    
    # Compile MLflow pipeline
    if compile_mlflow_pipeline():
        show_mlflow_instructions()
        show_mlflow_setup()
        show_example_scenarios()
        
        print(f"\n📁 Pipeline ready at: {os.getcwd()}")
        print("📄 File: aasist_mlflow_serving_pipeline.yaml")
        
        print("\n🚀 Next Steps:")
        print("1. Upload the YAML file to your Kubeflow dashboard")
        print("2. Create a new run with your desired parameters")
        print("3. Monitor the pipeline execution")
        print("4. Access your deployed model via the serving endpoint")
        
    else:
        print("\n❌ Pipeline compilation failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 