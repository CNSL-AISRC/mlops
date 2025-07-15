#!/usr/bin/env python3
"""
Simple runner script for AASIST Demo Pipeline
Compiles the pipeline and provides instructions for running
"""
import os
import sys
from pathlib import Path

def compile_demo_pipeline():
    """Compile the demo pipeline"""
    print("🔧 Compiling AASIST Demo Pipeline...")
    
    try:
        # Import and compile the demo pipeline
        from kubeflow_pipeline_demo import aasist_demo_pipeline
        import kfp
        
        # Compile pipeline
        kfp.compiler.Compiler().compile(
            aasist_demo_pipeline, 
            'aasist_demo_pipeline.yaml'
        )
        
        print("✅ Demo pipeline compiled successfully!")
        print(f"📄 Generated: aasist_demo_pipeline.yaml")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have kubeflow-pipelines installed:")
        print("pip install kfp")
        return False
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False

def show_demo_instructions():
    """Show instructions for running the demo"""
    print("\n" + "="*60)
    print("🎯 AASIST DEMO PIPELINE - QUICK START GUIDE")
    print("="*60)
    
    print("\n📋 What this demo includes:")
    print("  ✓ Mock dataset creation (fast, no large downloads)")
    print("  ✓ Pretrained model loading simulation")
    print("  ✓ Evaluation with realistic metrics")
    print("  ✓ MLflow integration and logging")
    print("  ✓ Comprehensive reporting")
    
    print("\n⚡ Expected runtime: 2-5 minutes")
    print("💾 Resource requirements: Minimal (CPU only)")
    
    print("\n🚀 How to run:")
    print("1. Upload 'aasist_demo_pipeline.yaml' to your Kubeflow dashboard")
    print("2. Create a new experiment (or use existing)")
    print("3. Create a new run with these parameters:")
    print("   • dataset_url: 'mock://demo_dataset'")
    print("   • config_name: 'AASIST' (or 'AASIST-L')")
    print("   • model_name: 'my_aasist_demo'")
    
    print("\n📊 Pipeline steps:")
    print("  1️⃣  Dataset Preparation (mock data)")
    print("  2️⃣  Pretrained Model Loading")
    print("  3️⃣  Model Evaluation (simulated)")
    print("  4️⃣  MLflow Logging")
    print("  5️⃣  Report Generation")
    
    print("\n🔄 For production training, use:")
    print("  • kubeflow_pipeline_production.py (full training)")
    print("  • distributed_main.py (multi-GPU training)")
    
    print("\n" + "="*60)

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

def main():
    """Main function"""
    print("🤖 AASIST Demo Pipeline Runner")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues before proceeding")
        sys.exit(1)
    
    # Compile demo pipeline
    if compile_demo_pipeline():
        show_demo_instructions()
        
        # Ask if user wants to see other pipeline options
        print("\n🔧 Other available pipelines:")
        print("  • Full training: python kubeflow_pipeline_production.py")
        print("  • Local distributed: python distributed_main.py --help")
        
        print(f"\n📁 All files are ready in: {os.getcwd()}")
        
    else:
        print("\n❌ Pipeline compilation failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 