#!/usr/bin/env python3
"""
Simple runner script for AASIST MLflow + Serving Pipeline
Compiles and runs the pipeline using KFP client
"""
import os
import sys
import kfp
from pathlib import Path

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

def run_mlflow_pipeline(operation="upload_and_serve", 
                       model_path="/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth",
                       model_name="aasist_production",
                       model_version="1.0",
                       model_stage="Production",
                       config_name="AASIST"):
    """Run the MLflow serving pipeline using KFP client"""
    print("🚀 Running AASIST MLflow + Serving Pipeline...")
    
    try:
        from kubeflow_pipeline_mlflow_serving import aasist_mlflow_serving_pipeline
        
        # Create KFP client
        client = kfp.Client()
        
        # Compile pipeline
        print("🔧 Compiling pipeline...")
        kfp.compiler.Compiler().compile(
            aasist_mlflow_serving_pipeline, 
            'upload.yaml'
        )
        print("✅ Pipeline compiled successfully!")
        
        # Run pipeline with simplified naming to avoid collation issues
        print(f"🚀 Starting pipeline run with operation: {operation}")
        
        import time
        import random
        
        # Try extremely simple names to avoid collation issues
        simple_names = [
            f"run{random.randint(1000, 9999)}",  # run1234
            f"r{random.randint(100000, 999999)}",  # r123456
            f"a{int(time.time()) % 100000}",  # a12345
            f"test{random.randint(100, 999)}"  # test123
        ]
        
        last_error = None
        for i, run_name in enumerate(simple_names):
            try:
                print(f"🔄 Attempt {i+1}/4: Using run name '{run_name}'")
                run = client.create_run_from_pipeline_func(
                    aasist_mlflow_serving_pipeline, 
                    arguments={
                        'operation': operation,
                        'model_path': model_path,
                        'model_name': model_name,
                        'model_version': model_version,
                        'model_stage': model_stage,
                        'config_name': config_name
                    },
                    enable_caching=False,
                    run_name=run_name
                )
                print(f"✅ Success with run name: {run_name}")
                break
            except Exception as e:
                last_error = e
                print(f"❌ Failed with '{run_name}': {str(e)[:100]}...")
                if i < len(simple_names) - 1:
                    print(f"🔄 Trying next name...")
                    continue
                else:
                    print("❌ All name attempts failed. This is a Kubeflow database collation issue.")
                    print("💡 Possible solutions:")
                    print("   1. Ask your admin to fix MySQL collation in Kubeflow database")
                    print("   2. Try running without explicit run names")
                    print("   3. Use a different Kubeflow installation")
                    
                    # Last attempt without explicit run name
                    print("🔄 Final attempt: No explicit run name...")
                    try:
                        run = client.create_run_from_pipeline_func(
                            aasist_mlflow_serving_pipeline, 
                            arguments={
                                'operation': operation,
                                'model_path': model_path,
                                'model_name': model_name,
                                'model_version': model_version,
                                'model_stage': model_stage,
                                'config_name': config_name
                            },
                            enable_caching=False
                            # No run_name parameter - let Kubeflow generate it
                        )
                        print("✅ Success without explicit run name!")
                        break
                    except Exception as final_e:
                        print(f"❌ Final attempt also failed: {str(final_e)[:100]}...")
                        raise final_e
        
        print(f"✅ Pipeline started successfully!")
        print(f"📊 Run ID: {run.run_id}")
        print(f"🔗 View in dashboard: {client._get_url_prefix()}/pipeline/#/runs/details/{run.run_id}")
        
        return run
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have kubeflow-pipelines installed:")
        print("pip install kfp mlflow")
        return None
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return None

def show_usage_examples():
    """Show usage examples for different scenarios"""
    print("\n" + "="*70)
    print("📚 USAGE EXAMPLES")
    print("="*70)
    
    print("\n🔧 Example 1: Upload and Serve (Default)")
    print("  python run_mlflow_serving.py")
    print("  # Uses local trained model and serves it")
    
    print("\n🔧 Example 2: Upload Only")
    print("  python run_mlflow_serving.py --operation upload_only --model_path /path/to/model.pth")
    
    print("\n🔧 Example 3: Serve Existing Model")
    print("  python run_mlflow_serving.py --operation serve_only --model_name my_model --model_version latest")
    
    print("\n🔧 Example 4: Custom Parameters")
    print("  python run_mlflow_serving.py --operation upload_and_serve \\")
    print("                               --model_path https://example.com/model.pth \\")
    print("                               --model_name custom_aasist \\")
    print("                               --model_version 2.0 \\")
    print("                               --model_stage Staging")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AASIST MLflow + Serving Pipeline Runner")
    parser.add_argument("--operation", default="upload_and_serve", 
                       choices=["upload_only", "serve_only", "upload_and_serve"],
                       help="Operation mode")
    parser.add_argument("--model_path", 
                       default="/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth",
                       help="Path to local model file")
    parser.add_argument("--model_name", default="aasist_production",
                       help="MLflow model name")
    parser.add_argument("--model_version", default="1.0",
                       help="Model version")
    parser.add_argument("--model_stage", default="Production",
                       help="MLflow model stage")
    parser.add_argument("--config_name", default="AASIST",
                       choices=["AASIST", "AASIST-L"],
                       help="Model configuration")
    
    args = parser.parse_args()
    
    print("🤖 AASIST MLflow + Serving Pipeline Runner")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues before proceeding")
        sys.exit(1)
    
    print(f"\n📋 Pipeline Parameters:")
    print(f"  • Operation: {args.operation}")
    print(f"  • Model Path: {args.model_path}")
    print(f"  • Model Name: {args.model_name}")
    print(f"  • Model Version: {args.model_version}")
    print(f"  • Model Stage: {args.model_stage}")
    print(f"  • Config: {args.config_name}")
    
    # Run pipeline
    run = run_mlflow_pipeline(
        operation=args.operation,
        model_path=args.model_path,
        model_name=args.model_name,
        model_version=args.model_version,
        model_stage=args.model_stage,
        config_name=args.config_name
    )
    
    if run:
        print("\n🎉 Pipeline execution initiated successfully!")
        print(f"⏱️  Monitor progress in the Kubeflow dashboard")
        
        if args.operation in ["serve_only", "upload_and_serve"]:
            print(f"🔗 Once complete, your model will be served at:")
            print(f"   http://aasist-serving-{args.model_name.lower().replace('_', '-')}.kubeflow.svc.cluster.local")
        
        show_usage_examples()
    else:
        print("\n❌ Pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 