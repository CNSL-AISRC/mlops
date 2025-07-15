#!/usr/bin/env python3
"""
Simple runner script for AASIST Demo Pipeline
Compiles and runs the demo pipeline using KFP client
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
        "kubeflow_pipeline_demo.py",
        "main.py",
        "models/AASIST.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"Missing file: {file}")
    
    # Check if KFP is available
    try:
        import kfp
        print(f"  ✓ Kubeflow Pipelines: {kfp.__version__}")
    except ImportError:
        issues.append("kubeflow-pipelines not installed (pip install kfp)")
    
    # Check MLflow
    try:
        import mlflow
        print(f"  ✓ MLflow: {mlflow.__version__}")
    except ImportError:
        issues.append("mlflow not installed (pip install mlflow)")
    
    if issues:
        print("❌ Environment issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✅ Environment looks good!")
        return True

def run_demo_pipeline(dataset_url="mock://demo_dataset", 
                     config_name="AASIST", 
                     model_name="aasist_demo"):
    """Run the demo pipeline using KFP client"""
    print("🚀 Running AASIST Demo Pipeline...")
    
    try:
        from kubeflow_pipeline_demo import aasist_demo_pipeline
        
        # Create KFP client
        client = kfp.Client()
        
        # Compile pipeline
        print("🔧 Compiling demo pipeline...")
        kfp.compiler.Compiler().compile(
            aasist_demo_pipeline, 
            'aasist_demo_pipeline.yaml'
        )
        print("✅ Demo pipeline compiled successfully!")
        
        # Run pipeline
        print(f"🚀 Starting demo pipeline run...")
        run = client.create_run_from_pipeline_func(
            aasist_demo_pipeline, 
            arguments={
                'dataset_url': dataset_url,
                'config_name': config_name,
                'model_name': model_name
            },
            enable_caching=False
        )
        
        print(f"✅ Demo pipeline started successfully!")
        print(f"📊 Run ID: {run.run_id}")
        print(f"🔗 View in dashboard: {client._get_url_prefix()}/pipeline/#/runs/details/{run.run_id}")
        
        return run
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have kubeflow-pipelines installed:")
        print("pip install kfp")
        return None
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return None

def show_demo_info():
    """Show information about the demo"""
    print("\n" + "="*60)
    print("🎯 AASIST DEMO PIPELINE - INFORMATION")
    print("="*60)
    
    print("\n📋 What this demo includes:")
    print("  ✓ Mock dataset creation (fast, no large downloads)")
    print("  ✓ Pretrained model loading simulation")
    print("  ✓ Evaluation with realistic metrics")
    print("  ✓ MLflow integration and logging")
    print("  ✓ HTTP serving API with health checks")
    print("  ✓ Comprehensive reporting")
    
    print("\n⚡ Expected runtime: 3-5 minutes")
    print("💾 Resource requirements: Minimal (CPU only)")
    
    print("\n📊 Pipeline steps:")
    print("  1️⃣  Dataset Preparation (mock data)")
    print("  2️⃣  Pretrained Model Loading")
    print("  3️⃣  Model Evaluation (simulated)")
    print("  4️⃣  MLflow Logging & Model Upload")
    print("  5️⃣  HTTP Serving Deployment")
    print("  6️⃣  Service Testing & Validation")
    print("  7️⃣  Report Generation")
    
    print("\n🔧 Configuration options:")
    print("  • dataset_url: 'mock://demo_dataset' (demo) or real URL")
    print("  • config_name: 'AASIST' or 'AASIST-L'") 
    print("  • model_name: Name for MLflow model registry")

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    
    print("\n🔧 Example 1: Default Demo")
    print("  python run_demo.py")
    print("  # Runs with mock data and AASIST config")
    
    print("\n🔧 Example 2: Different Config")
    print("  python run_demo.py --config AASIST-L --model_name my_large_model")
    
    print("\n🔧 Example 3: Custom Dataset")
    print("  python run_demo.py --dataset_url https://example.com/dataset.zip")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AASIST Demo Pipeline Runner")
    parser.add_argument("--dataset_url", default="mock://demo_dataset",
                       help="Dataset URL (use 'mock://demo_dataset' for demo)")
    parser.add_argument("--config", dest="config_name", default="AASIST",
                       choices=["AASIST", "AASIST-L"],
                       help="Model configuration")
    parser.add_argument("--model_name", default="aasist_demo",
                       help="MLflow model name")
    
    args = parser.parse_args()
    
    print("🤖 AASIST Demo Pipeline Runner")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues before proceeding")
        sys.exit(1)
    
    print(f"\n📋 Demo Parameters:")
    print(f"  • Dataset URL: {args.dataset_url}")
    print(f"  • Config: {args.config_name}")
    print(f"  • Model Name: {args.model_name}")
    
    # Show demo info
    show_demo_info()
    
    # Run demo pipeline
    run = run_demo_pipeline(
        dataset_url=args.dataset_url,
        config_name=args.config_name,
        model_name=args.model_name
    )
    
    if run:
        print("\n🎉 Demo pipeline execution initiated successfully!")
        print(f"⏱️  Monitor progress in the Kubeflow dashboard")
        print(f"🔗 Once complete, check the serving endpoint and reports")
        
        show_usage_examples()
        
        print("\n🔄 For production training, use:")
        print("  • python run_mlflow_serving.py (MLflow + serving)")
        print("  • Full training pipeline (if available)")
        
    else:
        print("\n❌ Demo pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 