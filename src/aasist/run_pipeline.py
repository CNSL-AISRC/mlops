#!/usr/bin/env python3
"""
Simple script to run the AASIST Anti-spoofing Pipeline

This script demonstrates how to compile and run the complete AASIST pipeline
with different configurations.
"""

import json
import argparse
from aasist_pipeline import aasist_pipeline
from kfp import compiler
from kfp_manager import KFPClientManager
from dotenv import load_dotenv
import os
import json

load_dotenv()


def compile_pipeline_only():
    """Just compile the pipeline without running it"""
    print("Compiling AASIST pipeline...")
    
    compiler.Compiler().compile(aasist_pipeline, 'aasist_pipeline.yaml')
    print("✅ Pipeline compiled successfully: aasist_pipeline.yaml")
    print("You can now upload this YAML file to the KFP UI or run it programmatically.")

def run_demo_pipeline(config):
    """Run a demo version of the pipeline with small dataset and quick training"""
    print("Running AASIST demo pipeline...")
    
    config["num_epochs"] = 1  # Quick training
    config["batch_size"] = 4  # Small batch size
    config_json = json.dumps(config)
    
    # Compile the pipeline
    compiler.Compiler().compile(aasist_pipeline, 'aasist_demo_pipeline.yaml')
    
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
            'config_json': config_json,
            'device': 'cuda'  # Change to 'cpu' if no GPU available
        },
        namespace="admin",
        experiment_name="aasist-demo-experiment",
    )
    
    print(f"✅ Demo pipeline run created: {run.run_id}")
    print(f"🔗 View run: {os.getenv('KFP_API_URL')}/runs/details/{run.run_id}")

def run_production_pipeline():
    """Run the full production pipeline with complete dataset and longer training"""
    print("Running AASIST production pipeline...")
    
    # Get default configuration and modify for production

    config["num_epochs"] = 20  # More training epochs
    config["batch_size"] = 8   # Larger batch size
    config_json = json.dumps(config)
    
    # Compile the pipeline
    compiler.Compiler().compile(aasist_pipeline, 'aasist_production_pipeline.yaml')
    
    # Create KFP client
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "true").lower() == "true",
        dex_username=os.getenv("DEX_USERNAME"),
        dex_password=os.getenv("DEX_PASSWORD"),
        dex_auth_type=os.getenv("DEX_AUTH_TYPE", "local"),
    )
    
    kfp_client = kfp_client_manager.create_kfp_client()
    
    # Run the pipeline with production parameters
    run = kfp_client.create_run_from_pipeline_package(
        'aasist_production_pipeline.yaml',
        arguments={
            'config_json': config_json,
            'device': 'cuda'
        },
        namespace="admin",
        experiment_name="aasist-production-experiment",
    )
    
    print(f"✅ Production pipeline run created: {run.run_id}")
    print(f"🔗 View run: {os.getenv('KFP_API_URL')}/runs/details/{run.run_id}")

def run_custom_pipeline(config_file: str):
    """Run pipeline with custom configuration from file"""
    print(f"Running AASIST pipeline with config: {config_file}")
    
    # Load custom configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    config_json = json.dumps(config)
    
    # Compile the pipeline
    compiler.Compiler().compile(aasist_pipeline, 'aasist_custom_pipeline.yaml')
    
    # Create KFP client
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KFP_API_URL"),
        skip_tls_verify=os.getenv("KFP_SKIP_TLS_VERIFY", "true").lower() == "true",
        dex_username=os.getenv("DEX_USERNAME"),
        dex_password=os.getenv("DEX_PASSWORD"),
        dex_auth_type=os.getenv("DEX_AUTH_TYPE", "local"),
    )
    
    kfp_client = kfp_client_manager.create_kfp_client()
    
    # Extract pipeline parameters from config
    pipeline_args = {
        'config_json': config_json,
        'device': config.get('device', 'cuda')
    }
    
    # Run the pipeline
    run = kfp_client.create_run_from_pipeline_package(
        'aasist_custom_pipeline.yaml',
        arguments=pipeline_args,
        namespace="admin",
        experiment_name="aasist-custom-experiment",
    )
    
    print(f"✅ Custom pipeline run created: {run.run_id}")
    print(f"🔗 View run: {os.getenv('KFP_API_URL')}/runs/details/{run.run_id}")

def main():
    parser = argparse.ArgumentParser(description="Run AASIST Anti-spoofing Pipeline")
    parser.add_argument(
        '--mode',
        choices=['compile', 'demo', 'production', 'custom', 'default'],
        help='Pipeline execution mode',
        default='demo'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (required for custom mode)'
    )
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read()) 
    
    print("🎵 AASIST Anti-spoofing Pipeline Runner 🎵")
    print("=" * 50)
    
    # Check environment variables
    required_env_vars = ['KFP_API_URL', 'DEX_USERNAME', 'DEX_PASSWORD']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars and args.mode != 'compile':
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set them in your .env file or environment.")
        return 1
    
    try:
        if args.mode == 'compile':
            compile_pipeline_only()
        elif args.mode == 'demo':
            run_demo_pipeline(config)
        elif args.mode == 'production':
            run_production_pipeline()
        elif args.mode == 'custom':
            if not args.config:
                print("❌ Custom mode requires --config argument")
                return 1
            run_custom_pipeline(args.config)
        # elif args.mode == 'default':
        #     compile_and_run_pipeline()
        
        print("\n✅ Pipeline execution completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 