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

def run_demo_pipeline(dataset_url="http://10.5.110.131:8080/LA.zip", 
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
            'demo.yaml'
        )
        print("✅ Demo pipeline compiled successfully!")
        
        # Run pipeline with simplified naming to avoid collation issues
        print(f"🚀 Starting demo pipeline run...")
        
        # Try with simple experiment name first
        import time
        timestamp = str(int(time.time()))
        simple_run_name = f"aasist-demo-{timestamp}"
        
        try:
            run = client.create_run_from_pipeline_func(
                aasist_demo_pipeline, 
                arguments={
                    'dataset_url': dataset_url,
                    'config_name': config_name,
                    'model_name': model_name
                },
                enable_caching=False,
                run_name=simple_run_name
            )
        except Exception as e:
            print(f"⚠️  First attempt failed (likely DB collation issue): {str(e)[:100]}...")
            print("🔄 Trying with minimal run name...")
            
            # Fallback with even simpler name
            simple_name = f"demo{timestamp[-6:]}"
            run = client.create_run_from_pipeline_func(
                aasist_demo_pipeline, 
                arguments={
                    'dataset_url': dataset_url,
                    'config_name': config_name,
                    'model_name': model_name
                },
                enable_caching=False,
                run_name=simple_name
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
    print("  ✓ Real dataset download and preparation")
    print("  ✓ AASIST model training/evaluation")
    print("  ✓ MLflow integration and logging")
    print("  ✓ HTTP serving API deployment")
    print("  ✓ Comprehensive reporting")
    
    print("\n⚡ Expected runtime: 10-15 minutes")
    print("💾 Resource requirements: CPU/GPU recommended")
    
    print("\n📊 Pipeline steps:")
    print("  1️⃣  Dataset Download & Preparation")
    print("  2️⃣  Model Training/Evaluation")
    print("  3️⃣  MLflow Logging & Model Upload")
    print("  4️⃣  HTTP Serving Deployment")
    print("  5️⃣  Service Testing & Validation")
    print("  6️⃣  Report Generation")
    
    print("\n🔧 Configuration options:")
    print("  • dataset_url: Real dataset URL")
    print("  • config_name: 'AASIST' or 'AASIST-L'") 
    print("  • model_name: Name for MLflow model registry")

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    
    print("\n🔧 Example 1: Default Demo")
    print("  python run_demo.py")
    print("  # Uses real LA dataset with AASIST config")
    
    print("\n🔧 Example 2: Different Config")
    print("  python run_demo.py --config AASIST-L --model_name my_large_model")
    
    print("\n🔧 Example 3: Custom Dataset")
    print("  python run_demo.py --dataset_url https://example.com/dataset.zip")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AASIST Demo Pipeline Runner")
    parser.add_argument("--dataset_url", default="http://10.5.110.131:8080/LA.zip",
                       help="Dataset URL")
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