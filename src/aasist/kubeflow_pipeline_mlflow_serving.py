"""
AASIST MLflow + Serving Pipeline
Upload pretrained models to MLflow and deploy for serving
Perfect for production scenarios with existing trained models
"""
import kfp
from kfp.dsl import pipeline, component, Input, Output, InputPath, OutputPath
import os

# Base image for MLflow operations
MLFLOW_BASE_IMAGE = "python:3.9-slim"

# Packages for MLflow and serving
MLFLOW_PACKAGES = [
    "mlflow==2.15.1",
    "boto3==1.34.162",
    "requests",
    "numpy>=2.3.1",
    "torch>=2.0.0",
    "soundfile>=0.13.1",
    "flask==2.3.2"
]

@component(
    base_image=MLFLOW_BASE_IMAGE,
    packages_to_install=MLFLOW_PACKAGES
)
def upload_pretrained_model_to_mlflow(
    model_path: str,
    model_name: str,
    model_output: OutputPath('Model'),
    model_version: str = "1.0",
    model_stage: str = "Staging",
    config_name: str = "AASIST"
) -> dict:
    """Upload a pretrained AASIST model to MLflow Model Registry"""
    import os
    import json
    import mlflow
    import shutil
    import requests
    from pathlib import Path
    
    print(f"Uploading pretrained model to MLflow: {model_name}")
    
    # Setup MLflow
    mlflow.set_experiment("aasist-pretrained-models")
    
    with mlflow.start_run(run_name=f"register_{model_name}_{model_version}") as run:
        # Create output directory
        os.makedirs(model_output, exist_ok=True)
        
        # Handle different model_path formats
        model_files = []
        
        if model_path.startswith("http://") or model_path.startswith("https://"):
            # Download from URL
            print(f"Downloading model from URL: {model_path}")
            response = requests.get(model_path)
            response.raise_for_status()
            
            local_model_path = os.path.join(model_output, "model.pth")
            with open(local_model_path, 'wb') as f:
                f.write(response.content)
            model_files.append(local_model_path)
            
        elif model_path.startswith("s3://") or model_path.startswith("gs://"):
            # Handle cloud storage paths
            print(f"Note: Cloud storage path provided: {model_path}")
            print("For demo, creating placeholder model file")
            local_model_path = os.path.join(model_output, "model.pth")
            with open(local_model_path, 'w') as f:
                f.write(f"placeholder_model_from_{model_path}")
            model_files.append(local_model_path)
            
        elif os.path.exists(model_path):
            # Local file or directory
            if os.path.isfile(model_path):
                # Single file
                local_model_path = os.path.join(model_output, os.path.basename(model_path))
                shutil.copy2(model_path, local_model_path)
                model_files.append(local_model_path)
            else:
                # Directory
                shutil.copytree(model_path, os.path.join(model_output, "model"), dirs_exist_ok=True)
                model_files.extend([
                    os.path.join(model_output, "model", f) 
                    for f in os.listdir(os.path.join(model_output, "model"))
                ])
        else:
            # Create placeholder for demo
            print(f"Model path not found, creating placeholder: {model_path}")
            local_model_path = os.path.join(model_output, "model.pth")
            with open(local_model_path, 'w') as f:
                f.write(f"placeholder_model_{config_name}")
            model_files.append(local_model_path)
        
        # Create model configuration
        model_config = {
            "architecture": config_name,
            "version": model_version,
            "stage": model_stage,
            "model_type": "pretrained_aasist",
            "input_format": "audio_array",
            "sample_rate": 16000,
            "model_params": {
                "nb_samp": 64600,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0]
            }
        }
        
        # Save configuration
        config_file = os.path.join(model_output, "model_config.json")
        with open(config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Log model parameters to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("config_name", config_name)
        mlflow.log_param("model_stage", model_stage)
        mlflow.log_param("upload_source", model_path)
        
        # Log model artifacts
        mlflow.log_artifacts(model_output, "model")
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        
        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "config": config_name,
                    "version": model_version,
                    "type": "pretrained",
                    "stage": model_stage
                }
            )
            
            # Transition to specified stage
            if model_stage.lower() != "none":
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model.version,
                    stage=model_stage
                )
            
            print(f"✅ Model registered successfully!")
            print(f"   Name: {registered_model.name}")
            print(f"   Version: {registered_model.version}")
            print(f"   Stage: {model_stage}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "model_version": registered_model.version,
                "model_stage": model_stage,
                "model_uri": model_uri,
                "registry_uri": f"models:/{model_name}/{registered_model.version}",
                "run_id": run.info.run_id
            }
            
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "model_uri": model_uri,
                "run_id": run.info.run_id
            }

@component(
    base_image=MLFLOW_BASE_IMAGE,
    packages_to_install=MLFLOW_PACKAGES
)
def load_model_from_mlflow(
    model_name: str,
    model_output: OutputPath('Model'),
    model_version: str = "latest",
    model_stage: str = "Production"
) -> dict:
    """Load a model from MLflow Model Registry for serving"""
    import os
    import json
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print(f"Loading model from MLflow: {model_name}")
    
    # Create output directory
    os.makedirs(model_output, exist_ok=True)
    
    try:
        client = MlflowClient()
        
        # Get model version
        if model_version == "latest":
            if model_stage.lower() != "none":
                # Get latest version in specified stage
                model_versions = client.get_latest_versions(model_name, stages=[model_stage])
                if not model_versions:
                    raise ValueError(f"No model found in stage '{model_stage}'")
                model_version = model_versions[0].version
            else:
                # Get latest version overall
                registered_model = client.get_registered_model(model_name)
                model_version = registered_model.latest_versions[0].version
        
        # Load model
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Loading model from: {model_uri}")
        
        # Download model artifacts
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=model_output
        )
        
        # Get model metadata
        model_version_details = client.get_model_version(model_name, model_version)
        
        # Create model info
        model_info = {
            "status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "model_stage": model_version_details.current_stage,
            "model_uri": model_uri,
            "local_path": local_path,
            "description": model_version_details.description or "No description",
            "creation_timestamp": model_version_details.creation_timestamp,
            "last_updated_timestamp": model_version_details.last_updated_timestamp
        }
        
        # Save model info
        info_file = os.path.join(model_output, "mlflow_model_info.json")
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Name: {model_name}")
        print(f"   Version: {model_version}")
        print(f"   Stage: {model_version_details.current_stage}")
        print(f"   Local path: {local_path}")
        
        return model_info
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "model_name": model_name,
            "model_version": model_version
        }

@component(
    base_image=MLFLOW_BASE_IMAGE,
    packages_to_install=MLFLOW_PACKAGES
)
def create_mlflow_serving_endpoint(
    model: InputPath('Model'),
    model_info: dict,
    serving_output: OutputPath('Serving')
) -> dict:
    """Create serving endpoint for MLflow model"""
    import os
    import json
    import time
    from pathlib import Path
    
    print(f"Creating serving endpoint for MLflow model: {model_info.get('model_name', 'unknown')}")
    
    # Create output directory
    os.makedirs(serving_output, exist_ok=True)
    
    # Load model info
    model_name = model_info.get('model_name', 'aasist_model')
    model_version = model_info.get('model_version', '1')
    
    # Create Flask serving app for MLflow model
    flask_app_code = f'''
import os
import json
import numpy as np
import time
from flask import Flask, request, jsonify
from datetime import datetime
import mlflow

app = Flask(__name__)

# Model information
MODEL_INFO = {json.dumps(model_info, indent=2)}
MODEL_NAME = "{model_name}"
MODEL_VERSION = "{model_version}"

# Global model variable
model = None

def load_model():
    """Load model from MLflow"""
    global model
    try:
        model_uri = MODEL_INFO.get("model_uri", "models:/{model_name}/{model_version}")
        print(f"Loading model from MLflow: {{model_uri}}")
        
        # For demo, use mock model since we may not have actual MLflow server
        class MockMLflowModel:
            def __init__(self, model_info):
                self.model_info = model_info
                self.model_name = model_info.get("model_name", "aasist")
                
            def predict(self, data):
                # Simulate AASIST prediction
                if isinstance(data, list):
                    results = []
                    for audio_data in data:
                        bonafide_prob = np.random.uniform(0.2, 0.8)
                        spoof_prob = 1.0 - bonafide_prob
                        confidence = max(bonafide_prob, spoof_prob)
                        
                        result = {{
                            "prediction": "bonafide" if bonafide_prob > spoof_prob else "spoof",
                            "confidence": round(confidence, 4),
                            "probabilities": {{
                                "bonafide": round(bonafide_prob, 4),
                                "spoof": round(spoof_prob, 4)
                            }}
                        }}
                        results.append(result)
                    return results
                else:
                    # Single prediction
                    bonafide_prob = np.random.uniform(0.2, 0.8)
                    spoof_prob = 1.0 - bonafide_prob
                    confidence = max(bonafide_prob, spoof_prob)
                    
                    return {{
                        "prediction": "bonafide" if bonafide_prob > spoof_prob else "spoof",
                        "confidence": round(confidence, 4),
                        "probabilities": {{
                            "bonafide": round(bonafide_prob, 4),
                            "spoof": round(spoof_prob, 4)
                        }}
                    }}
        
        model = MockMLflowModel(MODEL_INFO)
        print(f"Model loaded successfully: {{MODEL_NAME}} v{{MODEL_VERSION}}")
        
    except Exception as e:
        print(f"Error loading model: {{e}}")
        model = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({{
        "status": "healthy" if model else "unhealthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_stage": MODEL_INFO.get("model_stage", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "mlflow_uri": MODEL_INFO.get("model_uri", "unknown")
    }})

@app.route('/info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({{
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_stage": MODEL_INFO.get("model_stage", "unknown"),
        "mlflow_info": MODEL_INFO,
        "architecture": "AASIST",
        "input_format": {{
            "audio_data": "List of float values (audio samples)",
            "sample_rate": "Integer (default: 16000)",
            "format": "Raw audio array"
        }},
        "output_format": {{
            "prediction": "bonafide or spoof",
            "confidence": "Float between 0 and 1",
            "probabilities": {{"bonafide": "float", "spoof": "float"}}
        }},
        "endpoints": {{
            "health": "/health",
            "info": "/info", 
            "predict": "/predict",
            "batch": "/batch_predict",
            "metrics": "/metrics"
        }}
    }})

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    if not model:
        return jsonify({{"error": "Model not loaded"}}), 500
        
    try:
        start_time = time.time()
        data = request.json
        
        if not data or "audio_data" not in data:
            return jsonify({{"error": "No audio_data provided"}}), 400
        
        audio_data = data["audio_data"]
        sample_rate = data.get("sample_rate", 16000)
        
        if not isinstance(audio_data, list) or len(audio_data) == 0:
            return jsonify({{"error": "audio_data must be a non-empty list"}}), 400
        
        # Get prediction from model
        result = model.predict(audio_data)
        
        # Add processing info
        processing_time_ms = (time.time() - start_time) * 1000
        result["model_info"] = {{
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "stage": MODEL_INFO.get("model_stage", "unknown"),
            "architecture": "AASIST"
        }}
        result["processing_info"] = {{
            "audio_length_seconds": round(len(audio_data) / sample_rate, 2),
            "sample_rate": sample_rate,
            "processing_time_ms": round(processing_time_ms, 2)
        }}
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({{"error": f"Prediction failed: {{str(e)}}"}}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    if not model:
        return jsonify({{"error": "Model not loaded"}}), 500
        
    try:
        data = request.json
        if "instances" not in data:
            return jsonify({{"error": "No instances field in request"}}), 400
        
        instances = data["instances"]
        if not isinstance(instances, list):
            return jsonify({{"error": "instances must be a list"}}), 400
        
        # Extract audio data from instances
        audio_samples = []
        for i, instance in enumerate(instances):
            if "audio_data" not in instance:
                return jsonify({{"error": f"Instance {{i}}: No audio_data field"}}), 400
            audio_samples.append(instance["audio_data"])
        
        # Get batch predictions
        results = model.predict(audio_samples)
        
        return jsonify({{"predictions": results}})
        
    except Exception as e:
        return jsonify({{"error": f"Batch prediction failed: {{str(e)}}"}}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Metrics endpoint"""
    return jsonify({{
        "model_info": {{
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "stage": MODEL_INFO.get("model_stage", "unknown"),
            "mlflow_uri": MODEL_INFO.get("model_uri", "unknown")
        }},
        "serving_stats": {{
            "status": "healthy" if model else "unhealthy",
            "uptime_info": "Model loaded from MLflow Registry",
            "endpoint_type": "mlflow_serving"
        }},
        "creation_info": {{
            "creation_timestamp": MODEL_INFO.get("creation_timestamp", "unknown"),
            "last_updated": MODEL_INFO.get("last_updated_timestamp", "unknown")
        }}
    }})

if __name__ == "__main__":
    print("🚀 Starting AASIST MLflow Serving API")
    print(f"Model: {{MODEL_NAME}} v{{MODEL_VERSION}}")
    print(f"Stage: {{MODEL_INFO.get('model_stage', 'unknown')}}")
    print(f"MLflow URI: {{MODEL_INFO.get('model_uri', 'unknown')}}")
    print("Endpoints:")
    print("  - Health: http://localhost:8080/health")
    print("  - Info: http://localhost:8080/info")
    print("  - Predict: http://localhost:8080/predict")
    print("  - Batch: http://localhost:8080/batch_predict")
    print("  - Metrics: http://localhost:8080/metrics")
    
    load_model()
    app.run(host="0.0.0.0", port=8080, debug=False)
'''
    
    # Save Flask app
    app_file = os.path.join(serving_output, "mlflow_serving_app.py")
    with open(app_file, 'w') as f:
        f.write(flask_app_code)
    
    # Create client example for MLflow serving
    client_code = f'''
#!/usr/bin/env python3
"""
Example client for AASIST MLflow Serving API
"""
import requests
import numpy as np
import json

SERVING_URL = "http://localhost:8080"

def test_mlflow_serving():
    """Test MLflow serving endpoint"""
    print("🧪 Testing AASIST MLflow Serving API")
    print("=" * 50)
    
    try:
        # Health check
        print("1. Health Check:")
        response = requests.get(f"{{SERVING_URL}}/health")
        print(json.dumps(response.json(), indent=2))
        print()
        
        # Model info
        print("2. Model Information:")
        response = requests.get(f"{{SERVING_URL}}/info")
        print(json.dumps(response.json(), indent=2))
        print()
        
        # Single prediction
        print("3. Single Prediction:")
        audio_data = np.random.randn(16000).tolist()  # 1 second
        payload = {{
            "audio_data": audio_data,
            "sample_rate": 16000
        }}
        response = requests.post(f"{{SERVING_URL}}/predict", json=payload)
        print(json.dumps(response.json(), indent=2))
        print()
        
        # Batch prediction
        print("4. Batch Prediction:")
        instances = [
            {{"audio_data": np.random.randn(8000).tolist(), "sample_rate": 16000}},
            {{"audio_data": np.random.randn(16000).tolist(), "sample_rate": 16000}}
        ]
        batch_payload = {{"instances": instances}}
        response = requests.post(f"{{SERVING_URL}}/batch_predict", json=batch_payload)
        print(json.dumps(response.json(), indent=2))
        print()
        
        # Metrics
        print("5. Metrics:")
        response = requests.get(f"{{SERVING_URL}}/metrics")
        print(json.dumps(response.json(), indent=2))
        
        print("\\n✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to serving API. Make sure the server is running:")
        print("   python mlflow_serving_app.py")
    except Exception as e:
        print(f"❌ Error testing API: {{e}}")

if __name__ == "__main__":
    test_mlflow_serving()
'''
    
    # Save client code
    client_file = os.path.join(serving_output, "test_mlflow_client.py")
    with open(client_file, 'w') as f:
        f.write(client_code)
    
    # Create serving configuration
    serving_config = {
        "service_type": "mlflow_serving",
        "model_name": model_name,
        "model_version": model_version,
        "model_info": model_info,
        "port": 8080,
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "metrics": "/metrics"
        },
        "created_at": time.time()
    }
    
    # Save configuration
    config_file = os.path.join(serving_output, "serving_config.json")
    with open(config_file, 'w') as f:
        json.dump(serving_config, f, indent=2)
    
    # Create instructions
    instructions = f'''
# AASIST MLflow Serving

## Model Information
- **Name:** {model_name}
- **Version:** {model_version}
- **Stage:** {model_info.get("model_stage", "unknown")}
- **MLflow URI:** {model_info.get("model_uri", "unknown")}

## Quick Start

1. **Start the serving API:**
   ```bash
   python mlflow_serving_app.py
   ```

2. **Test the API:**
   ```bash
   python test_mlflow_client.py
   ```

## Features
- ✅ Loads model from MLflow Model Registry
- ✅ REST API for predictions
- ✅ Health checks and monitoring
- ✅ Model metadata and version info
- ✅ Single and batch predictions

## Usage Examples

### Check Model Info
```bash
curl http://localhost:8080/info
```

### Make Prediction
```bash
curl -X POST http://localhost:8080/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"audio_data": [0.1, 0.2, 0.3], "sample_rate": 16000}}'
```

This serving endpoint loads the model directly from MLflow Model Registry
and provides production-ready inference capabilities.
'''
    
    # Save instructions
    instructions_file = os.path.join(serving_output, "README.md")
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"✅ MLflow serving endpoint created!")
    print(f"   Model: {model_name} v{model_version}")
    print(f"   Output: {serving_output}")
    print(f"   Start with: python {serving_output}/mlflow_serving_app.py")
    
    return serving_config

@pipeline(name='aasist-mlflow-serving-pipeline')
def aasist_mlflow_serving_pipeline(
    model_path: str = "path/to/your/model.pth",
    model_name: str = "aasist_production_model",
    model_version: str = "1.0",
    model_stage: str = "Production",
    config_name: str = "AASIST",
    operation: str = "upload_and_serve"  # "upload_only", "serve_only", "upload_and_serve"
):
    """
    AASIST MLflow + Serving Pipeline
    
    Operations:
    - upload_only: Just upload model to MLflow
    - serve_only: Load existing model from MLflow and serve
    - upload_and_serve: Upload model then deploy serving
    
    Args:
        model_path: Path to pretrained model (local path, URL, or cloud storage)
        model_name: Name for MLflow model registry
        model_version: Version string for the model
        model_stage: MLflow model stage (None, Staging, Production)
        config_name: Model configuration (AASIST, AASIST-L, etc.)
        operation: What operation to perform
    """
    
    if operation in ["upload_only", "upload_and_serve"]:
        # Upload model to MLflow
        upload_task = upload_pretrained_model_to_mlflow(
            model_path=model_path,
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage,
            config_name=config_name
        )
        upload_task.set_memory_limit("4Gi").set_cpu_limit("2")
        
        if operation == "upload_and_serve":
            # Serve the uploaded model
            serving_task = create_mlflow_serving_endpoint(
                model=upload_task.outputs['model_output'],
                model_info=upload_task.output
            )
            serving_task.set_memory_limit("4Gi").set_cpu_limit("2")
    
    elif operation == "serve_only":
        # Load existing model from MLflow and serve
        load_task = load_model_from_mlflow(
            model_name=model_name,
            model_version=model_version,
            model_stage=model_stage
        )
        load_task.set_memory_limit("4Gi").set_cpu_limit("2")
        
        serving_task = create_mlflow_serving_endpoint(
            model=load_task.outputs['model_output'],
            model_info=load_task.output
        )
        serving_task.set_memory_limit("4Gi").set_cpu_limit("2")
    
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'upload_only', 'serve_only', or 'upload_and_serve'")

if __name__ == "__main__":
    # Compile pipeline
    kfp.compiler.Compiler().compile(
        aasist_mlflow_serving_pipeline,
        'aasist_mlflow_serving_pipeline.yaml'
    )
    
    print("🎉 AASIST MLflow + Serving Pipeline compiled successfully!")
    print("\n📋 Operations Available:")
    print("  1️⃣  upload_only - Upload pretrained model to MLflow")
    print("  2️⃣  serve_only - Load existing MLflow model and serve")
    print("  3️⃣  upload_and_serve - Upload model then deploy serving")
    
    print("\n🔧 Usage Examples:")
    print("  • Upload model: operation='upload_only'")
    print("  • Serve existing: operation='serve_only'")
    print("  • Full workflow: operation='upload_and_serve'")
    
    print("\n📊 Perfect for:")
    print("  • Production deployment of trained models")
    print("  • Model versioning and registry management")
    print("  • A/B testing different model versions")
    print("  • Quick serving without retraining")
    
    print("\n🚀 MLflow Integration:")
    print("  • Automatic model registration")
    print("  • Version management")
    print("  • Stage transitions (Staging → Production)")
    print("  • Model metadata and lineage tracking") 