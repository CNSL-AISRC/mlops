#!/usr/bin/env python3
"""
KServe-based Serving Runner for AASIST Models
Deploy AASIST models using KServe for persistent serving
"""
import os
import sys
import kfp
from pathlib import Path
import argparse

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

def run_kserve_serving_pipeline(model_path, config_name, service_name, namespace="admin"):
    """Run the KServe-based serving pipeline using KFP client"""
    print("🚀 Running AASIST KServe Serving Pipeline...")
    
    try:
        from kubeflow_pipeline_serving_fixed import aasist_kserve_serving_pipeline
        
        # Create KFP client
        client = kfp.Client()
        
        # Create or get experiment for multi-user mode
        try:
            experiment_name = "aasist-kserve-experiments"
            experiment = client.create_experiment(
                name=experiment_name,
                description="AASIST KServe serving experiments"
            )
            experiment_id = experiment.id
            print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            if "already exists" in str(e).lower():
                experiment = client.get_experiment(experiment_name=experiment_name)
                experiment_id = experiment.id
                print(f"✅ Using existing experiment: {experiment_name} (ID: {experiment_id})")
            else:
                print(f"⚠️  Could not create/get experiment: {e}")
                print("🔄 Using default experiment")
                experiment_id = None
        
        # Compile pipeline
        print("🔧 Compiling KServe serving pipeline...")
        kfp.compiler.Compiler().compile(
            aasist_kserve_serving_pipeline,
            'aasist_kserve_serving_pipeline.yaml'
        )
        print("✅ KServe serving pipeline compiled successfully!")
        
        # Create run
        print("🚀 Starting KServe serving pipeline...")
        run_args = {
            'model_path': model_path,
            'config_name': config_name,
            'service_name': service_name,
            'namespace': namespace
        }
        
        run = client.create_run_from_pipeline_func(
            aasist_kserve_serving_pipeline,
            arguments=run_args,
            experiment_id=experiment_id,
            enable_caching=False
        )
        
        print(f"✅ KServe serving pipeline started successfully!")
        print(f"📊 Run ID: {run.run_id}")
        
        if hasattr(run, 'run_url'):
            print(f"🔗 View in dashboard: {run.run_url}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Pipeline import error: {e}")
        print("💡 Make sure kubeflow_pipeline_serving_fixed.py exists")
        return False
    except Exception as e:
        print(f"❌ Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    print("\n🔧 Example 1: Default (Local model)")
    print("  python run_serving_kserve.py")
    print("  # Uses local trained model in admin namespace")
    
    print("\n🔧 Example 2: Custom Model")
    print("  python run_serving_kserve.py --model_path /path/to/model.pth --service_name my-model")
    
    print("\n🔧 Example 3: Different Config & Namespace")
    print("  python run_serving_kserve.py --config AASIST-L --namespace kubeflow --service_name large-model")
    
    print("\n🔧 Example 4: Custom Service Name")
    print("  python run_serving_kserve.py --service_name aasist-v2")

def show_deployment_info(service_name, namespace):
    """Show deployment information"""
    print(f"\n🎉 KServe deployment process initiated successfully!")
    print(f"⏱️  Monitor progress in the Kubeflow dashboard")
    print(f"\n📋 Service Details:")
    print(f"  • Service Name: {service_name}")
    print(f"  • Namespace: {namespace}")
    print(f"  • Expected URL: http://{service_name}-direct.{namespace}.svc.cluster.local:5000")
    
    print(f"\n🔗 Once complete, test with:")
    print(f"  python test.py")
    
    print(f"\n🔧 Manual checks:")
    print(f"  kubectl get inferenceservices -n {namespace}")
    print(f"  kubectl get services -n {namespace} | grep {service_name}")
    print(f"  kubectl get pods -n {namespace} | grep {service_name}")

def main():
    """Main entry point"""
    print("🤖 AASIST KServe Serving Pipeline Runner")
    print("=" * 45)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deploy AASIST model using KServe")
    parser.add_argument("--model_path", 
                       default="/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth",
                       help="Path to AASIST model file")
    parser.add_argument("--config", dest="config_name", 
                       default="AASIST", choices=["AASIST", "AASIST-L"],
                       help="Model configuration name")
    parser.add_argument("--service_name", 
                       default="aasist-serving",
                       help="Name for the serving service")
    parser.add_argument("--namespace", 
                       default="admin",
                       help="Kubernetes namespace for deployment")
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print(f"\n📋 KServe Serving Parameters:")
    print(f"  • Model Path: {args.model_path}")
    print(f"  • Config: {args.config_name}")
    print(f"  • Service Name: {args.service_name}")
    print(f"  • Namespace: {args.namespace}")
    
    print("\n" + "=" * 60)
    print("🎯 AASIST KSERVE SERVING PIPELINE")
    print("=" * 60)
    
    print(f"\n📋 What this pipeline does:")
    print(f"  ✓ Creates persistent KServe InferenceService")
    print(f"  ✓ Deploys HTTP serving API with proper endpoints")
    print(f"  ✓ Creates Kubernetes services for access")
    print(f"  ✓ Tests deployment after creation")
    
    print(f"\n⚡ Expected runtime: 3-5 minutes")
    print(f"💾 Resource requirements: 1-2 CPUs, 2-4GB RAM")
    
    print(f"\n📊 Pipeline steps:")
    print(f"  1️⃣  Create KServe InferenceService")
    print(f"  2️⃣  Deploy serving container") 
    print(f"  3️⃣  Create access services")
    print(f"  4️⃣  Test all endpoints")
    
    print(f"\n🔧 Configuration options:")
    print(f"  • model_path: Path to your trained model")
    print(f"  • config_name: 'AASIST' or 'AASIST-L'")
    print(f"  • service_name: Name for the serving service")
    print(f"  • namespace: Kubernetes namespace")
    
    # Run pipeline
    success = run_kserve_serving_pipeline(
        model_path=args.model_path,
        config_name=args.config_name,
        service_name=args.service_name,
        namespace=args.namespace
    )
    
    if success:
        show_deployment_info(args.service_name, args.namespace)
        show_usage_examples()
        
        print("\n🔄 For full training workflow, use:")
        print("  • python run_demo.py (full demo with dataset)")
        print("  • python run_mlflow_serving.py (MLflow integration)")
        
    else:
        print("\n❌ KServe serving pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 