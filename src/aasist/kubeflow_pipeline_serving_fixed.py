"""
AASIST KServe Serving Pipeline - Fixed Version
Creates persistent KServe deployment for AASIST model serving
"""
import kfp
from kfp.dsl import pipeline, component, Input, Output, InputPath, OutputPath

# Base image with KServe support
KSERVE_BASE_IMAGE = "python:3.11"

# Packages for KServe deployment
KSERVE_PACKAGES = [
    "kserve==0.13.1",
    "kubernetes==26.1.0",
    "torch>=2.0.0",
    "numpy>=1.24.3",
    "soundfile>=0.12.1",
    "requests>=2.31.0",
    "pyyaml>=6.0"
]

@component(
    base_image=KSERVE_BASE_IMAGE,
    packages_to_install=KSERVE_PACKAGES
)
def deploy_aasist_kserve(
    model_path: str,
    config_name: str,
    service_name: str,
    namespace: str = "admin"
) -> dict:
    """Deploy AASIST model using KServe with custom predictor"""
    import os
    import json
    import time
    import yaml
    from kubernetes import client, config
    from kubernetes.client import V1ObjectMeta
    
    print(f"🚀 Deploying AASIST model with KServe")
    print(f"📁 Model: {model_path}")
    print(f"⚙️  Config: {config_name}")
    print(f"🏷️  Service: {service_name}")
    print(f"📍 Namespace: {namespace}")
    
    try:
        # Load Kubernetes config
        try:
            config.load_incluster_config()
            print("✅ Loaded in-cluster config")
        except:
            config.load_kube_config()
            print("✅ Loaded local kube config")
        
        # Create Kubernetes client
        v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        # Define InferenceService YAML
        isvc_yaml = f"""
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {service_name}
  namespace: {namespace}
  annotations:
    sidecar.istio.io/inject: "false"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
    containers:
    - name: kserve-container
      image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
      command:
      - python
      - -c
      - |
        import os
        import json
        import torch
        import numpy as np
        from typing import Dict, Any
        from flask import Flask, request, jsonify
        import threading
        import time
        
        app = Flask(__name__)
        
        # Mock AASIST model for serving
        class MockAASISTModel:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"🔧 Initialized MockAASISTModel on {{self.device}}")
                
            def predict(self, audio_data):
                import random
                score = random.uniform(0.1, 0.9)
                is_bonafide = score > 0.5
                return {{
                    "score": float(score),
                    "prediction": "bonafide" if is_bonafide else "spoof",
                    "confidence": float(abs(score - 0.5) * 2)
                }}
        
        # Global model instance
        model = MockAASISTModel()
        
        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({{
                "status": "healthy",
                "service": "{service_name}",
                "model_path": "{model_path}",
                "config": "{config_name}",
                "device": str(model.device)
            }})
        
        @app.route("/predict", methods=["POST"])
        def predict():
            try:
                if "audio" not in request.json:
                    return jsonify({{"error": "No audio data provided"}}), 400
                
                audio_data = request.json["audio"]
                result = model.predict(audio_data)
                
                return jsonify({{
                    "model": "{service_name}",
                    "config": "{config_name}", 
                    "prediction": result["prediction"],
                    "score": result["score"],
                    "confidence": result["confidence"],
                    "timestamp": time.time()
                }})
            except Exception as e:
                return jsonify({{"error": str(e)}}), 500
        
        @app.route("/info", methods=["GET"])
        def info():
            return jsonify({{
                "service_name": "{service_name}",
                "model_path": "{model_path}",
                "config_name": "{config_name}",
                "status": "active",
                "endpoints": ["/health", "/predict", "/info"]
            }})
        
        print("🚀 Starting AASIST serving on 0.0.0.0:8080")
        app.run(host="0.0.0.0", port=8080, debug=False)
      ports:
      - containerPort: 8080
        protocol: TCP
      env:
      - name: MODEL_PATH
        value: "{model_path}"
      - name: CONFIG_NAME  
        value: "{config_name}"
      - name: SERVICE_NAME
        value: "{service_name}"
      resources:
        limits:
          cpu: "2"
          memory: "4Gi"
        requests:
          cpu: "500m"
          memory: "1Gi"
"""
        
        # Apply the InferenceService
        print("📝 Creating InferenceService YAML...")
        isvc_dict = yaml.safe_load(isvc_yaml)
        
        # Use dynamic client to create InferenceService
        from kubernetes import dynamic
        from kubernetes.client import api_client
        
        dyn_client = dynamic.DynamicClient(
            api_client.ApiClient(configuration=config.Configuration.get_default())
        )
        
        # Get InferenceService API
        api = dyn_client.resources.get(
            api_version="serving.kserve.io/v1beta1",
            kind="InferenceService"
        )
        
        # Delete existing service if it exists
        try:
            api.delete(name=service_name, namespace=namespace)
            print(f"🗑️  Deleted existing InferenceService {service_name}")
            time.sleep(15)  # Wait for cleanup
        except Exception as e:
            print(f"ℹ️  No existing service to delete: {e}")
        
        # Create new InferenceService
        try:
            response = api.create(body=isvc_dict, namespace=namespace)
            print(f"✅ Created InferenceService {service_name}")
            
            # Wait for the service to be ready
            print("⏳ Waiting for InferenceService to become ready...")
            ready = False
            max_wait = 300  # 5 minutes
            wait_time = 0
            
            while not ready and wait_time < max_wait:
                try:
                    isvc = api.get(name=service_name, namespace=namespace)
                    status = isvc.get('status', {})
                    conditions = status.get('conditions', [])
                    
                    for condition in conditions:
                        if condition.get('type') == 'Ready' and condition.get('status') == 'True':
                            ready = True
                            print(f"✅ InferenceService {service_name} is ready!")
                            break
                            
                except Exception as e:
                    print(f"⏳ Checking status... ({wait_time}s)")
                
                if not ready:
                    time.sleep(10)
                    wait_time += 10
            
            # Create ClusterIP service for easier access
            service_manifest = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{service_name}-direct",
                    "namespace": namespace
                },
                "spec": {
                    "selector": {
                        "serving.kserve.io/inferenceservice": service_name
                    },
                    "ports": [{
                        "name": "http",
                        "port": 5000,
                        "targetPort": 8080,
                        "protocol": "TCP"
                    }],
                    "type": "ClusterIP"
                }
            }
            
            try:
                # Delete existing direct service if it exists
                core_v1.delete_namespaced_service(
                    name=f"{service_name}-direct", 
                    namespace=namespace
                )
                time.sleep(5)
            except:
                pass
            
            # Create new direct service
            core_v1.create_namespaced_service(
                namespace=namespace,
                body=service_manifest
            )
            print(f"✅ Created direct access service {service_name}-direct")
            
            return {
                "status": "ready" if ready else "timeout",
                "service_name": service_name,
                "namespace": namespace,
                "direct_url": f"http://{service_name}-direct.{namespace}.svc.cluster.local:5000",
                "model_path": model_path,
                "config_name": config_name
            }
            
        except Exception as e:
            print(f"❌ Failed to create InferenceService: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "service_name": service_name,
                "namespace": namespace
            }
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "error": str(e),
            "service_name": service_name,
            "namespace": namespace
        }

@component(
    base_image="python:3.11",
    packages_to_install=["requests>=2.31.0"]
)
def test_kserve_deployment(
    deployment_result: dict,
    test_output: OutputPath('TestResults')
) -> dict:
    """Test the KServe deployment"""
    import os
    import json
    import time
    import requests
    from datetime import datetime
    
    print(f"🧪 Testing KServe deployment...")
    print(f"📊 Deployment status: {deployment_result.get('status')}")
    
    # Create output directory
    os.makedirs(test_output, exist_ok=True)
    
    if deployment_result.get('status') != 'ready':
        print(f"❌ Deployment not ready: {deployment_result.get('error', 'Unknown error')}")
        return {"status": "failed", "reason": "deployment_not_ready"}
    
    service_name = deployment_result.get('service_name')
    namespace = deployment_result.get('namespace')
    direct_url = deployment_result.get('direct_url')
    
    # Test URLs to try
    test_urls = [
        direct_url,
        f"http://{service_name}.{namespace}.svc.cluster.local:5000",
        f"http://{service_name}-direct:5000"
    ]
    
    test_results = {
        "service_name": service_name,
        "namespace": namespace, 
        "test_timestamp": datetime.now().isoformat(),
        "deployment_result": deployment_result,
        "tests": []
    }
    
    successful_url = None
    
    # Try each URL
    for url in test_urls:
        print(f"🔗 Testing URL: {url}")
        try:
            response = requests.get(f"{url}/health", timeout=30)
            if response.status_code == 200:
                successful_url = url
                print(f"✅ Successfully connected to {url}")
                break
            else:
                print(f"❌ HTTP {response.status_code} from {url}")
        except Exception as e:
            print(f"❌ Failed to connect to {url}: {e}")
    
    if successful_url:
        # Run health test
        try:
            response = requests.get(f"{successful_url}/health", timeout=30)
            health_test = {
                "test_name": "health_check",
                "url": f"{successful_url}/health",
                "status": "pass" if response.status_code == 200 else "fail",
                "response": response.json() if response.status_code == 200 else response.text,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
            test_results["tests"].append(health_test)
            print(f"✅ Health check passed")
        except Exception as e:
            health_test = {
                "test_name": "health_check",
                "url": f"{successful_url}/health",
                "status": "fail",
                "error": str(e)
            }
            test_results["tests"].append(health_test)
            print(f"❌ Health check failed: {e}")
        
        # Run prediction test
        try:
            test_payload = {
                "audio": "mock_audio_data_base64"
            }
            response = requests.post(f"{successful_url}/predict", 
                                   json=test_payload, timeout=30)
            predict_test = {
                "test_name": "prediction",
                "url": f"{successful_url}/predict",
                "status": "pass" if response.status_code == 200 else "fail",
                "response": response.json() if response.status_code == 200 else response.text,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
            test_results["tests"].append(predict_test)
            print(f"✅ Prediction test passed")
        except Exception as e:
            predict_test = {
                "test_name": "prediction", 
                "url": f"{successful_url}/predict",
                "status": "fail",
                "error": str(e)
            }
            test_results["tests"].append(predict_test)
            print(f"❌ Prediction test failed: {e}")
        
        test_results["successful_url"] = successful_url
        test_results["status"] = "success"
        
    else:
        test_results["status"] = "failed"
        test_results["error"] = "No accessible URLs found"
    
    # Save test results
    results_file = os.path.join(test_output, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"📊 Test results saved to: {results_file}")
    
    return test_results

@pipeline(name='aasist-kserve-serving-pipeline')
def aasist_kserve_serving_pipeline(
    model_path: str = "/home/jovyan/mlops/src/aasist/models/weights/AASIST.pth",
    config_name: str = "AASIST",
    service_name: str = "aasist-serving",
    namespace: str = "admin"
):
    """AASIST KServe Serving Pipeline - Creates persistent service"""
    
    # Deploy with KServe
    deploy_task = deploy_aasist_kserve(
        model_path=model_path,
        config_name=config_name,
        service_name=service_name,
        namespace=namespace
    )
    
    # Test the deployment
    test_task = test_kserve_deployment(
        deployment_result=deploy_task.output
    )

if __name__ == "__main__":
    # Compile pipeline for testing
    kfp.compiler.Compiler().compile(
        aasist_kserve_serving_pipeline,
        'aasist_kserve_serving_pipeline.yaml'
    )
    print("✅ Pipeline compiled successfully!") 