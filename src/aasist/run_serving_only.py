#!/usr/bin/env python3
"""
Simple Serving-Only Runner for AASIST Models
Deploy existing models for serving without dataset preparation
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
    
    # Check if KFP is available
    try:
        import kfp
        print(f"  ✓ Kubeflow Pipelines: {kfp.__version__}")
    except ImportError:
        issues.append("kubeflow-pipelines not installed (pip install kfp)")
    
    if issues:
        print("❌ Environment issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✅ Environment looks good!")
        return True

def run_serving_pipeline(model_path="/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth",
                        config_name="AASIST", 
                        service_name="aasist-serving"):
    """Run the serving-only pipeline using KFP client"""
    print("🚀 Running AASIST Serving-Only Pipeline...")
    
    try:
        from kubeflow_pipeline_serving_simple import aasist_simple_serving_pipeline
        
        # Create KFP client
        client = kfp.Client()
        
        # Compile pipeline
        print("🔧 Compiling serving pipeline...")
        kfp.compiler.Compiler().compile(
            aasist_simple_serving_pipeline, 
            'aasist_simple_serving.yaml'
        )
        print("✅ Serving pipeline compiled successfully!")
        
        # Run pipeline with simplified naming
        print(f"🚀 Starting serving pipeline...")
        
        import time
        timestamp = str(int(time.time()))
        simple_run_name = f"serve-{timestamp}"
        
        try:
            run = client.create_run_from_pipeline_func(
                aasist_simple_serving_pipeline, 
                arguments={
                    'model_path': model_path,
                    'config_name': config_name,
                    'service_name': service_name
                },
                enable_caching=False,
                run_name=simple_run_name
            )
        except Exception as e:
            print(f"⚠️  First attempt failed: {str(e)[:100]}...")
            print("🔄 Trying with minimal run name...")
            
            # Fallback with even simpler name
            simple_name = f"serve{timestamp[-6:]}"
            run = client.create_run_from_pipeline_func(
                aasist_simple_serving_pipeline, 
                arguments={
                    'model_path': model_path,
                    'config_name': config_name,
                    'service_name': service_name
                },
                enable_caching=False,
                run_name=simple_name
            )
        
        print(f"✅ Serving pipeline started successfully!")
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

def show_serving_info():
    """Show information about serving-only pipeline"""
    print("\n" + "="*60)
    print("🎯 AASIST SERVING-ONLY PIPELINE")
    print("="*60)
    
    print("\n📋 What this pipeline does:")
    print("  ✓ Loads your trained AASIST model")
    print("  ✓ Deploys HTTP serving API")
    print("  ✓ No dataset download or training")
    print("  ✓ Quick deployment for inference")
    
    print("\n⚡ Expected runtime: 2-3 minutes")
    print("💾 Resource requirements: Minimal (CPU only)")
    
    print("\n📊 Pipeline steps:")
    print("  1️⃣  Load Model")
    print("  2️⃣  Deploy Serving API")
    print("  3️⃣  Test API Endpoints")
    
    print("\n🔧 Configuration options:")
    print("  • model_path: Path to your trained model")
    print("  • config_name: 'AASIST' or 'AASIST-L'") 
    print("  • service_name: Name for the serving service")

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    
    print("\n🔧 Example 1: Default (Local model)")
    print("  python run_serving_only.py")
    print("  # Uses local trained model")
    
    print("\n🔧 Example 2: Custom Model")
    print("  python run_serving_only.py --model_path /path/to/model.pth --service_name my-model")
    
    print("\n🔧 Example 3: Different Config")
    print("  python run_serving_only.py --config AASIST-L --service_name large-model")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AASIST Serving-Only Pipeline Runner")
    parser.add_argument("--model_path", 
                       default="/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth",
                       help="Path to local model file")
    parser.add_argument("--config", dest="config_name", default="AASIST",
                       choices=["AASIST", "AASIST-L"],
                       help="Model configuration")
    parser.add_argument("--service_name", default="aasist-serving",
                       help="Name for the serving service")
    
    args = parser.parse_args()
    
    print("🤖 AASIST Serving-Only Pipeline Runner")
    print("=" * 45)
    
    # Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues before proceeding")
        sys.exit(1)
    
    print(f"\n📋 Serving Parameters:")
    print(f"  • Model Path: {args.model_path}")
    print(f"  • Config: {args.config_name}")
    print(f"  • Service Name: {args.service_name}")
    
    # Show serving info
    show_serving_info()
    
    # Run serving pipeline
    run = run_serving_pipeline(
        model_path=args.model_path,
        config_name=args.config_name,
        service_name=args.service_name
    )
    
    if run:
        print("\n🎉 Serving pipeline execution initiated successfully!")
        print(f"⏱️  Monitor progress in the Kubeflow dashboard")
        print(f"🔗 Once complete, your model will be served at:")
        print(f"   http://{args.service_name}.kubeflow.svc.cluster.local")
        
        show_usage_examples()
        
        print("\n🔄 For full training workflow, use:")
        print("  • python run_demo.py (full demo with dataset)")
        print("  • python run_mlflow_serving.py (MLflow integration)")
        
    else:
        print("\n❌ Serving pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 