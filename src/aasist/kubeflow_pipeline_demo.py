"""
AASIST Demo Kubeflow Pipeline
Simplified pipeline for demonstration purposes
- Downloads dataset
- Loads pretrained model
- Runs evaluation
- Logs results to MLflow
"""
import kfp
from kfp.dsl import pipeline, component, Input, Output, InputPath, OutputPath
import os

# Lightweight base image for demo
BASE_IMAGE = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"

# Minimal packages for demo
DEMO_PACKAGES = [
    "numpy>=2.3.1",
    "soundfile>=0.13.1", 
    "tqdm>=4.67.1",
    "mlflow==2.15.1",
    "requests",
    "zipfile36"
]

@component(
    base_image=BASE_IMAGE,
    packages_to_install=DEMO_PACKAGES
)
def download_demo_dataset(
    dataset_url: str,
    dataset_path: OutputPath('Dataset')
) -> str:
    """Download and prepare a small subset of ASVspoof2019 dataset for demo"""
    import os
    import requests
    import zipfile
    from pathlib import Path
    
    print(f"Downloading demo dataset from {dataset_url}")
    
    # Create dataset directory
    os.makedirs(dataset_path, exist_ok=True)
    
    # For demo, we can use a smaller subset or mock the dataset
    if "asvspoof" in dataset_url.lower():
        # Download the full dataset but we'll only use a subset for evaluation
        print("Downloading ASVspoof2019 dataset...")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        zip_path = os.path.join(dataset_path, "LA.zip")
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # Remove zip file to save space
        os.remove(zip_path)
        
        # Verify dataset structure
        la_path = os.path.join(dataset_path, "LA")
        if os.path.exists(la_path):
            print(f"Dataset successfully prepared at {dataset_path}")
            return f"Full dataset prepared with {len(os.listdir(la_path))} items"
        else:
            raise ValueError(f"Expected LA directory not found in {dataset_path}")
    else:
        # For demo with mock data
        print("Creating mock dataset for demo...")
        mock_la_path = os.path.join(dataset_path, "LA")
        os.makedirs(mock_la_path, exist_ok=True)
        
        # Create mock directory structure
        for split in ["train", "dev", "eval"]:
            split_dir = os.path.join(mock_la_path, f"ASVspoof2019_LA_{split}")
            os.makedirs(split_dir, exist_ok=True)
            
            # Create a few mock audio files
            for i in range(10):
                mock_file = os.path.join(split_dir, f"LA_T_{i:07d}.flac")
                with open(mock_file, 'w') as f:
                    f.write("mock_audio_data")
        
        # Create mock protocol files
        protocol_dir = os.path.join(mock_la_path, "ASVspoof2019_LA_cm_protocols")
        os.makedirs(protocol_dir, exist_ok=True)
        
        for split in ["trn", "dev", "eval"]:
            protocol_file = os.path.join(protocol_dir, f"ASVspoof2019.LA.cm.{split}.trl.txt")
            with open(protocol_file, 'w') as f:
                for i in range(10):
                    f.write(f"LA LA_T_{i:07d} - A01 bonafide\n")
        
        print(f"Mock dataset created at {dataset_path}")
        return "Mock dataset created for demo"

@component(
    base_image=BASE_IMAGE,
    packages_to_install=DEMO_PACKAGES
)
def load_pretrained_aasist_model(
    config_name: str,
    model_output: OutputPath('Model'),
    run_name: str = "aasist_pretrained_demo"
) -> str:
    """Load pretrained AASIST model and prepare it for evaluation"""
    import os
    import json
    import mlflow
    from pathlib import Path
    
    print(f"Loading pretrained {config_name} model...")
    
    # Setup MLflow tracking
    mlflow.set_experiment("aasist-demo")
    
    with mlflow.start_run(run_name=run_name) as run:
        # Create output directory
        os.makedirs(model_output, exist_ok=True)
        
        # Model configurations
        model_configs = {
            "AASIST": {
                "architecture": "AASIST",
                "nb_samp": 64600,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0],
                "pretrained_weights_url": "https://github.com/clovaai/aasist/releases/download/v1.0/AASIST.pth"
            },
            "AASIST-L": {
                "architecture": "AASIST",  # Same architecture, different weights
                "nb_samp": 64600,
                "first_conv": 128,
                "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                "gat_dims": [64, 32],
                "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                "temperatures": [2.0, 2.0, 100.0, 100.0],
                "pretrained_weights_url": "https://github.com/clovaai/aasist/releases/download/v1.0/AASIST-L.pth"
            }
        }
        
        model_config = model_configs.get(config_name, model_configs["AASIST"])
        
        # Log model configuration
        mlflow.log_param("config_name", config_name)
        mlflow.log_param("architecture", model_config["architecture"])
        mlflow.log_param("model_type", "pretrained")
        
        # For demo purposes, create a mock model file
        # In a real scenario, you would download the actual pretrained weights
        model_file = os.path.join(model_output, "pretrained_model.pth")
        
        try:
            # Try to download actual pretrained weights
            import requests
            weights_url = model_config.get("pretrained_weights_url")
            if weights_url:
                print(f"Downloading pretrained weights from {weights_url}")
                response = requests.get(weights_url, timeout=30)
                if response.status_code == 200:
                    with open(model_file, 'wb') as f:
                        f.write(response.content)
                    print("Pretrained weights downloaded successfully")
                else:
                    raise Exception(f"Failed to download weights: {response.status_code}")
            else:
                raise Exception("No pretrained weights URL provided")
                
        except Exception as e:
            print(f"Could not download pretrained weights: {e}")
            print("Creating mock model weights for demo...")
            # Create a mock model file
            with open(model_file, 'w') as f:
                f.write(f"mock_pretrained_weights_for_{config_name}")
        
        # Save model configuration
        config_file = os.path.join(model_output, "model_config.json")
        with open(config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Log model artifacts to MLflow
        mlflow.log_artifacts(model_output, "pretrained_model")
        
        # Return model URI
        model_uri = f"{run.info.artifact_uri}/pretrained_model"
        print(f"Pretrained model prepared: {model_uri}")
        return model_uri

@component(
    base_image=BASE_IMAGE,
    packages_to_install=DEMO_PACKAGES
)
def evaluate_pretrained_model_demo(
    dataset: InputPath('Dataset'),
    model: InputPath('Model'),
    config_name: str,
    evaluation_output: OutputPath('Evaluation')
) -> dict:
    """Run evaluation on pretrained model (simplified for demo)"""
    import os
    import json
    import random
    from pathlib import Path
    
    print(f"Running evaluation for {config_name} model...")
    
    # Create output directory
    os.makedirs(evaluation_output, exist_ok=True)
    
    # Check if we have the model files
    model_files = os.listdir(model)
    print(f"Model files found: {model_files}")
    
    # Check dataset structure
    dataset_path = os.path.join(dataset, "LA")
    if os.path.exists(dataset_path):
        dataset_contents = os.listdir(dataset_path)
        print(f"Dataset contents: {dataset_contents}")
    else:
        print("Dataset LA directory not found, using mock evaluation")
    
    # For demo purposes, simulate evaluation results
    # In a real scenario, this would run the actual AASIST evaluation code
    
    # Simulate realistic performance metrics based on model type
    if config_name == "AASIST":
        # These are the reported results from the paper
        simulated_eer = round(random.uniform(0.8, 0.9), 2)  # Around 0.83%
        simulated_tdcf = round(random.uniform(0.025, 0.030), 4)  # Around 0.0275
    elif config_name == "AASIST-L":
        # Slightly worse performance for the lightweight version
        simulated_eer = round(random.uniform(0.95, 1.05), 2)  # Around 0.99%
        simulated_tdcf = round(random.uniform(0.030, 0.035), 4)  # Around 0.0309
    else:
        # Default values
        simulated_eer = round(random.uniform(1.0, 2.0), 2)
        simulated_tdcf = round(random.uniform(0.03, 0.05), 4)
    
    # Create evaluation results
    results = {
        "model_name": config_name,
        "EER_percent": simulated_eer,
        "min_tDCF": simulated_tdcf,
        "evaluation_type": "demo_simulation",
        "dataset": "ASVspoof2019_LA_eval",
        "total_trials": 71237,  # Actual number of trials in ASVspoof2019 LA eval set
        "bonafide_trials": 7355,
        "spoof_trials": 63882,
        "status": "completed_demo"
    }
    
    # Add some additional demo metrics
    results["demo_metrics"] = {
        "processing_time_seconds": round(random.uniform(45, 120), 1),
        "model_size_mb": round(random.uniform(5, 15), 1),
        "inference_speed_ms_per_sample": round(random.uniform(2, 8), 2)
    }
    
    # Save detailed results
    results_file = os.path.join(evaluation_output, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create a summary report
    summary_file = os.path.join(evaluation_output, "evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"AASIST Model Evaluation Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Model: {config_name}\n")
        f.write(f"Equal Error Rate (EER): {simulated_eer}%\n")
        f.write(f"Minimum tandem Detection Cost Function (min t-DCF): {simulated_tdcf}\n")
        f.write(f"Dataset: ASVspoof2019 Logical Access (LA) evaluation set\n")
        f.write(f"Total trials: {results['total_trials']}\n")
        f.write(f"Processing time: {results['demo_metrics']['processing_time_seconds']} seconds\n")
        f.write(f"\nThis is a demo evaluation with simulated results.\n")
        f.write(f"In a production pipeline, actual audio processing and model inference would occur.\n")
    
    print(f"Evaluation completed!")
    print(f"EER: {simulated_eer}%, min t-DCF: {simulated_tdcf}")
    print(f"Results saved to {evaluation_output}")
    
    return results

@component(
    base_image=BASE_IMAGE,
    packages_to_install=DEMO_PACKAGES + ["boto3==1.34.162"]
)
def log_model_to_mlflow_demo(
    model: InputPath('Model'),
    evaluation: InputPath('Evaluation'),
    config_name: str,
    model_name: str = "aasist_demo_model"
) -> str:
    """Log the evaluated model to MLflow with results"""
    import os
    import json
    import mlflow
    from pathlib import Path
    
    print(f"Logging {config_name} model to MLflow...")
    
    # Setup MLflow
    mlflow.set_experiment("aasist-demo-models")
    
    with mlflow.start_run(run_name=f"{model_name}_{config_name}") as run:
        # Load evaluation results
        eval_results_file = os.path.join(evaluation, "evaluation_results.json")
        if os.path.exists(eval_results_file):
            with open(eval_results_file, 'r') as f:
                eval_results = json.load(f)
        else:
            eval_results = {"status": "no_evaluation_results"}
        
        # Log model parameters
        mlflow.log_param("model_name", config_name)
        mlflow.log_param("model_type", "pretrained_aasist")
        mlflow.log_param("dataset", "ASVspoof2019_LA")
        
        # Log evaluation metrics
        if "EER_percent" in eval_results:
            mlflow.log_metric("EER_percent", eval_results["EER_percent"])
        if "min_tDCF" in eval_results:
            mlflow.log_metric("min_tDCF", eval_results["min_tDCF"])
        
        # Log demo metrics
        if "demo_metrics" in eval_results:
            for metric, value in eval_results["demo_metrics"].items():
                mlflow.log_metric(metric, value)
        
        # Log model artifacts
        mlflow.log_artifacts(model, "model")
        mlflow.log_artifacts(evaluation, "evaluation")
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={"config": config_name, "type": "demo"}
            )
            print(f"Model registered in MLflow: {registered_model.name} version {registered_model.version}")
            return f"Model registered: {model_name} v{registered_model.version}"
        except Exception as e:
            print(f"Could not register model: {e}")
            return f"Model logged to MLflow run: {run.info.run_id}"

@component(
    base_image=BASE_IMAGE,
    packages_to_install=DEMO_PACKAGES
)
def generate_demo_report(
    evaluation: InputPath('Evaluation'),
    mlflow_info: str,
    config_name: str,
    report_output: OutputPath('Report')
) -> dict:
    """Generate a comprehensive demo report"""
    import os
    import json
    from datetime import datetime
    
    print("Generating demo report...")
    
    # Create output directory
    os.makedirs(report_output, exist_ok=True)
    
    # Load evaluation results
    eval_results_file = os.path.join(evaluation, "evaluation_results.json")
    if os.path.exists(eval_results_file):
        with open(eval_results_file, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {"status": "no_evaluation_results"}
    
    # Create comprehensive report
    report = {
        "demo_info": {
            "pipeline_name": "AASIST Demo Pipeline",
            "timestamp": datetime.now().isoformat(),
            "model_config": config_name,
            "status": "completed"
        },
        "model_performance": eval_results,
        "mlflow_info": mlflow_info,
        "pipeline_summary": {
            "steps_completed": [
                "Dataset Download",
                "Pretrained Model Loading", 
                "Model Evaluation",
                "MLflow Logging",
                "Report Generation"
            ],
            "total_runtime_estimate": "2-5 minutes",
            "resource_usage": "Minimal (demo mode)"
        }
    }
    
    # Save JSON report
    report_file = os.path.join(report_output, "demo_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable report
    readable_report = os.path.join(report_output, "demo_report.md")
    with open(readable_report, 'w') as f:
        f.write(f"# AASIST Demo Pipeline Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Model Configuration\n")
        f.write(f"- **Model:** {config_name}\n")
        f.write(f"- **Type:** Pretrained AASIST\n")
        f.write(f"- **Dataset:** ASVspoof2019 Logical Access\n\n")
        
        f.write(f"## Performance Results\n")
        if "EER_percent" in eval_results:
            f.write(f"- **Equal Error Rate (EER):** {eval_results['EER_percent']}%\n")
        if "min_tDCF" in eval_results:
            f.write(f"- **Minimum t-DCF:** {eval_results['min_tDCF']}\n")
        
        f.write(f"\n## Pipeline Execution\n")
        f.write(f"- ✅ Dataset Download & Preparation\n")
        f.write(f"- ✅ Pretrained Model Loading\n")
        f.write(f"- ✅ Model Evaluation\n")
        f.write(f"- ✅ MLflow Model Logging\n")
        f.write(f"- ✅ Report Generation\n")
        
        f.write(f"\n## MLflow Integration\n")
        f.write(f"- **Status:** {mlflow_info}\n")
        
        f.write(f"\n## Notes\n")
        f.write(f"This is a demonstration pipeline showcasing AASIST integration with Kubeflow.\n")
        f.write(f"For production use, replace with actual training and full evaluation.\n")
    
    print(f"Demo report generated at {report_output}")
    return report

@component(
    base_image=BASE_IMAGE,
    packages_to_install=DEMO_PACKAGES + ["flask==2.3.2"]
)
def deploy_model_demo_serving(
    model: InputPath('Model'),
    evaluation: InputPath('Evaluation'),
    config_name: str,
    serving_output: OutputPath('Serving')
) -> dict:
    """Deploy model for demo serving with simple HTTP endpoint"""
    import os
    import json
    import time
    import numpy as np
    from pathlib import Path
    
    print(f"Creating demo serving endpoint for {config_name}")
    
    # Create output directory
    os.makedirs(serving_output, exist_ok=True)
    
    # Load evaluation results for serving metadata
    eval_results_file = os.path.join(evaluation, "evaluation_results.json")
    if os.path.exists(eval_results_file):
        with open(eval_results_file, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {"EER_percent": 1.0, "min_tDCF": 0.05}
    
    # Create serving configuration
    serving_config = {
        "model_name": config_name,
        "model_path": model,
        "serving_port": 8080,
        "health_endpoint": "/health",
        "predict_endpoint": "/predict",
        "model_metadata": eval_results,
        "created_at": time.time(),
        "demo_mode": True
    }
    
    # Create Flask serving app
    flask_app_code = f'''
import os
import json
import numpy as np
import time
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Model metadata
MODEL_CONFIG = {json.dumps(serving_config, indent=2)}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({{
        "status": "healthy",
        "model": "{config_name}",
        "timestamp": datetime.now().isoformat(),
        "version": "demo-1.0"
    }})

@app.route('/info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({{
        "model_name": "{config_name}",
        "architecture": "AASIST",
        "performance": {{
            "EER_percent": {eval_results.get("EER_percent", 1.0)},
            "min_tDCF": {eval_results.get("min_tDCF", 0.05)}
        }},
        "input_format": {{
            "audio_data": "List of float values (audio samples)",
            "sample_rate": "Integer (default: 16000)",
            "format": "Raw audio array or base64 encoded"
        }},
        "output_format": {{
            "prediction": "bonafide or spoof",
            "confidence": "Float between 0 and 1",
            "probabilities": {{"bonafide": "float", "spoof": "float"}}
        }}
    }})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for audio anti-spoofing"""
    try:
        start_time = time.time()
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({{"error": "No JSON data provided"}}), 400
        
        # Check for audio data
        if "audio_data" not in data:
            return jsonify({{"error": "No audio_data field in request"}}), 400
        
        audio_data = data["audio_data"]
        sample_rate = data.get("sample_rate", 16000)
        
        # Validate audio data
        if not isinstance(audio_data, list) or len(audio_data) == 0:
            return jsonify({{"error": "audio_data must be a non-empty list"}}), 400
        
        # Simulate processing time based on audio length
        processing_time = len(audio_data) / sample_rate * 0.1  # 10% of audio duration
        time.sleep(min(processing_time, 0.5))  # Max 0.5 seconds for demo
        
        # Generate realistic demo prediction
        # Longer audio tends to be easier to classify (higher confidence)
        audio_length = len(audio_data) / sample_rate
        base_confidence = min(0.95, 0.7 + audio_length * 0.05)
        
        # Add some randomness
        bonafide_prob = np.random.beta(2, 2) * 0.6 + 0.2  # Between 0.2 and 0.8
        if np.random.random() > 0.5:  # Randomly flip to make it spoof
            bonafide_prob = 1.0 - bonafide_prob
        
        spoof_prob = 1.0 - bonafide_prob
        confidence = max(bonafide_prob, spoof_prob)
        
        # Ensure confidence is reasonable
        if confidence < 0.6:
            if bonafide_prob > spoof_prob:
                bonafide_prob = 0.7
                spoof_prob = 0.3
            else:
                bonafide_prob = 0.3
                spoof_prob = 0.7
            confidence = 0.7
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        result = {{
            "prediction": "bonafide" if bonafide_prob > spoof_prob else "spoof",
            "confidence": round(confidence, 4),
            "probabilities": {{
                "bonafide": round(bonafide_prob, 4),
                "spoof": round(spoof_prob, 4)
            }},
            "model_info": {{
                "name": "{config_name}",
                "architecture": "AASIST",
                "version": "demo-1.0"
            }},
            "processing_info": {{
                "audio_length_seconds": round(audio_length, 2),
                "sample_rate": sample_rate,
                "processing_time_ms": round(processing_time_ms, 2)
            }}
        }}
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({{"error": f"Prediction failed: {{str(e)}}"}}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.json
        if "instances" not in data:
            return jsonify({{"error": "No instances field in request"}}), 400
        
        instances = data["instances"]
        if not isinstance(instances, list):
            return jsonify({{"error": "instances must be a list"}}), 400
        
        results = []
        for i, instance in enumerate(instances):
            if "audio_data" not in instance:
                results.append({{"error": f"Instance {{i}}: No audio_data field"}})
                continue
            
            # Simulate individual prediction (simplified)
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
        
        return jsonify({{"predictions": results}})
        
    except Exception as e:
        return jsonify({{"error": f"Batch prediction failed: {{str(e)}}"}}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Basic metrics endpoint"""
    return jsonify({{
        "model_performance": {{
            "EER_percent": {eval_results.get("EER_percent", 1.0)},
            "min_tDCF": {eval_results.get("min_tDCF", 0.05)}
        }},
        "serving_stats": {{
            "uptime_seconds": time.time() - MODEL_CONFIG["created_at"],
            "requests_served": "demo_mode",
            "average_response_time_ms": "~100-200ms"
        }}
    }})

if __name__ == "__main__":
    print("🚀 Starting AASIST Demo Serving API")
    print(f"Model: {config_name}")
    print(f"Endpoints available:")
    print(f"  - Health: http://localhost:8080/health")
    print(f"  - Info: http://localhost:8080/info")
    print(f"  - Predict: http://localhost:8080/predict")
    print(f"  - Batch: http://localhost:8080/batch_predict")
    print(f"  - Metrics: http://localhost:8080/metrics")
    
    app.run(host="0.0.0.0", port=8080, debug=False)
'''
    
    # Save Flask app
    app_file = os.path.join(serving_output, "serving_app.py")
    with open(app_file, 'w') as f:
        f.write(flask_app_code)
    
    # Create example client code
    client_example = f'''
#!/usr/bin/env python3
"""
Example client for AASIST Demo Serving API
"""
import requests
import numpy as np
import json

# Serving endpoint
SERVING_URL = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{{SERVING_URL}}/health")
    print(f"Health check: {{response.json()}}")

def test_info():
    """Test model info endpoint"""
    response = requests.get(f"{{SERVING_URL}}/info")
    print(f"Model info: {{json.dumps(response.json(), indent=2)}}")

def test_prediction():
    """Test prediction with sample audio"""
    # Generate 1 second of random audio (16kHz)
    sample_audio = np.random.randn(16000).tolist()
    
    payload = {{
        "audio_data": sample_audio,
        "sample_rate": 16000
    }}
    
    response = requests.post(f"{{SERVING_URL}}/predict", json=payload)
    print(f"Prediction result: {{json.dumps(response.json(), indent=2)}}")

def test_batch_prediction():
    """Test batch prediction"""
    # Generate multiple audio samples
    instances = []
    for i in range(3):
        audio = np.random.randn(8000).tolist()  # 0.5 seconds each
        instances.append({{"audio_data": audio, "sample_rate": 16000}})
    
    payload = {{"instances": instances}}
    
    response = requests.post(f"{{SERVING_URL}}/batch_predict", json=payload)
    print(f"Batch prediction: {{json.dumps(response.json(), indent=2)}}")

if __name__ == "__main__":
    print("🧪 Testing AASIST Demo Serving API")
    print("="*50)
    
    try:
        test_health()
        print()
        
        test_info()
        print()
        
        test_prediction()
        print()
        
        test_batch_prediction()
        print()
        
        # Test metrics
        response = requests.get(f"{{SERVING_URL}}/metrics")
        print(f"Metrics: {{json.dumps(response.json(), indent=2)}}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to serving API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Error testing API: {{e}}")
'''
    
    # Save client example
    client_file = os.path.join(serving_output, "test_client.py")
    with open(client_file, 'w') as f:
        f.write(client_example)
    
    # Create serving instructions
    instructions = f'''
# AASIST Demo Serving Instructions

## Quick Start

1. **Start the serving API:**
   ```bash
   cd {serving_output}
   python serving_app.py
   ```

2. **Test the API:**
   ```bash
   # In another terminal
   python test_client.py
   ```

## API Endpoints

- **Health Check:** `GET /health`
- **Model Info:** `GET /info`
- **Single Prediction:** `POST /predict`
- **Batch Prediction:** `POST /batch_predict`
- **Metrics:** `GET /metrics`

## Example Usage

### Single Prediction
```python
import requests
import numpy as np

# Generate sample audio (1 second at 16kHz)
audio_data = np.random.randn(16000).tolist()

payload = {{
    "audio_data": audio_data,
    "sample_rate": 16000
}}

response = requests.post("http://localhost:8080/predict", json=payload)
result = response.json()
print(f"Prediction: {{result['prediction']}}")
print(f"Confidence: {{result['confidence']}}")
```

### Batch Prediction
```python
instances = [
    {{"audio_data": np.random.randn(16000).tolist(), "sample_rate": 16000}},
    {{"audio_data": np.random.randn(8000).tolist(), "sample_rate": 16000}}
]

payload = {{"instances": instances}}
response = requests.post("http://localhost:8080/batch_predict", json=payload)
```

## Model Performance
- **EER:** {eval_results.get("EER_percent", 1.0)}%
- **min t-DCF:** {eval_results.get("min_tDCF", 0.05)}

This is a demo serving endpoint with simulated predictions.
For production use, replace with actual AASIST model inference.
'''
    
    # Save instructions
    instructions_file = os.path.join(serving_output, "README.md")
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    # Save serving configuration
    config_file = os.path.join(serving_output, "serving_config.json")
    with open(config_file, 'w') as f:
        json.dump(serving_config, f, indent=2)
    
    print(f"Demo serving endpoint created at {serving_output}")
    print(f"Start with: python {serving_output}/serving_app.py")
    
    return serving_config

@component(
    base_image=BASE_IMAGE,
    packages_to_install=DEMO_PACKAGES + ["mlflow==2.15.1"]
)
def upload_demo_model_to_mlflow(
    model: InputPath('Model'),
    evaluation: InputPath('Evaluation'),
    model_name: str,
    config_name: str
) -> dict:
    """Upload demo model to MLflow Model Registry"""
    import os
    import json
    import mlflow
    from pathlib import Path
    
    print(f"Uploading demo model to MLflow: {model_name}")
    
    # Setup MLflow
    mlflow.set_experiment("aasist-demo-models")
    
    with mlflow.start_run(run_name=f"demo_{model_name}") as run:
        # Load evaluation results
        eval_results_file = os.path.join(evaluation, "evaluation_results.json")
        if os.path.exists(eval_results_file):
            with open(eval_results_file, 'r') as f:
                eval_results = json.load(f)
        else:
            eval_results = {"EER_percent": 1.0, "min_tDCF": 0.05}
        
        # Log model parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("config_name", config_name)
        mlflow.log_param("model_type", "demo_pretrained")
        mlflow.log_param("demo_mode", True)
        
        # Log evaluation metrics
        if "EER_percent" in eval_results:
            mlflow.log_metric("EER_percent", eval_results["EER_percent"])
        if "min_tDCF" in eval_results:
            mlflow.log_metric("min_tDCF", eval_results["min_tDCF"])
        
        # Log model artifacts
        mlflow.log_artifacts(model, "model")
        mlflow.log_artifacts(evaluation, "evaluation")
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        
        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "config": config_name,
                    "type": "demo",
                    "EER": str(eval_results.get("EER_percent", 1.0)),
                    "min_tDCF": str(eval_results.get("min_tDCF", 0.05))
                }
            )
            
            print(f"✅ Demo model registered in MLflow!")
            print(f"   Name: {registered_model.name}")
            print(f"   Version: {registered_model.version}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "model_version": registered_model.version,
                "model_uri": model_uri,
                "registry_uri": f"models:/{model_name}/{registered_model.version}",
                "run_id": run.info.run_id,
                "performance": eval_results
            }
            
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "model_uri": model_uri,
                "run_id": run.info.run_id
            }

@pipeline(name='aasist-demo-pipeline')
def aasist_demo_pipeline(
    dataset_url: str = "mock://demo_dataset",  # Use mock for faster demo
    config_name: str = "AASIST",
    model_name: str = "aasist_demo_model",
    include_serving: bool = True,  # Include serving endpoint
    include_mlflow_upload: bool = True  # New: Upload to MLflow Registry
):
    """
    AASIST Demo Pipeline - Complete MLOps demonstration
    
    Args:
        dataset_url: URL for dataset (use 'mock://demo_dataset' for quick demo)
        config_name: Model configuration (AASIST or AASIST-L)  
        model_name: Name for MLflow model registration
        include_serving: Whether to deploy demo serving endpoint
        include_mlflow_upload: Whether to upload model to MLflow Model Registry
    """
    
    # Step 1: Download and prepare dataset
    download_task = download_demo_dataset(dataset_url=dataset_url)
    
    # Step 2: Load pretrained model
    model_task = load_pretrained_aasist_model(config_name=config_name)
    
    # Step 3: Run evaluation
    eval_task = evaluate_pretrained_model_demo(
        dataset=download_task.outputs['dataset_path'],
        model=model_task.outputs['model_output'],
        config_name=config_name
    )
    
    # Step 4: Log to MLflow (basic logging)
    mlflow_task = log_model_to_mlflow_demo(
        model=model_task.outputs['model_output'],
        evaluation=eval_task.outputs['evaluation_output'],
        config_name=config_name,
        model_name=model_name
    )
    
    # Step 5: Upload to MLflow Model Registry (optional)
    if include_mlflow_upload:
        registry_task = upload_demo_model_to_mlflow(
            model=model_task.outputs['model_output'],
            evaluation=eval_task.outputs['evaluation_output'],
            model_name=model_name,
            config_name=config_name
        )
        registry_task.set_memory_limit("2Gi").set_cpu_limit("1")
    
    # Step 6: Deploy serving endpoint (optional)
    if include_serving:
        serving_task = deploy_model_demo_serving(
            model=model_task.outputs['model_output'],
            evaluation=eval_task.outputs['evaluation_output'],
            config_name=config_name
        )
        serving_task.set_memory_limit("2Gi").set_cpu_limit("1")
    
    # Step 7: Generate comprehensive demo report
    report_task = generate_demo_report(
        evaluation=eval_task.outputs['evaluation_output'],
        mlflow_info=mlflow_task.output,
        config_name=config_name
    )
    
    # Set reasonable resource limits for demo
    download_task.set_memory_limit("2Gi").set_cpu_limit("1")
    model_task.set_memory_limit("2Gi").set_cpu_limit("1")
    eval_task.set_memory_limit("4Gi").set_cpu_limit("2")
    mlflow_task.set_memory_limit("2Gi").set_cpu_limit("1")
    report_task.set_memory_limit("1Gi").set_cpu_limit("1")

if __name__ == "__main__":
    # Compile pipeline
    kfp.compiler.Compiler().compile(
        aasist_demo_pipeline, 
        'aasist_demo_pipeline.yaml'
    )
    
    print("🎉 AASIST Demo Pipeline compiled successfully!")
    print("\n📋 Quick Start:")
    print("1. Upload 'aasist_demo_pipeline.yaml' to Kubeflow")
    print("2. Create a new run with parameters:")
    print("   - dataset_url: 'mock://demo_dataset' (for quick demo)")
    print("   - config_name: 'AASIST' or 'AASIST-L'")
    print("   - model_name: 'your_model_name'")
    print("   - include_serving: true (to deploy demo API)")
    print("   - include_mlflow_upload: true (to register in MLflow)")
    print("\n⚡ Expected runtime: 4-8 minutes")
    print("🔧 Resources needed: Minimal (no GPU required for demo)")
    print("\n📊 Complete MLOps Pipeline includes:")
    print("   ✓ Demo dataset preparation")
    print("   ✓ Pretrained model loading") 
    print("   ✓ Model evaluation (simulated)")
    print("   ✓ MLflow experiment tracking")
    print("   ✓ MLflow Model Registry upload")
    print("   ✓ HTTP serving API deployment")
    print("   ✓ API testing and validation")
    print("   ✓ Comprehensive reporting")
    
    print("\n🚀 Serving API Features:")
    print("   • REST API for audio anti-spoofing")
    print("   • Health checks and monitoring")
    print("   • Single and batch predictions")
    print("   • Model performance metrics")
    print("   • Example client code")
    
    print("\n📊 MLflow Integration:")
    print("   • Automatic experiment tracking")
    print("   • Model Registry management")
    print("   • Version control and lineage")
    print("   • Performance metrics logging")
    
    print("\n🔗 For production workflows:")
    print("   • Use kubeflow_pipeline_mlflow_serving.py")
    print("   • Upload existing models: operation='upload_only'")
    print("   • Serve from MLflow: operation='serve_only'")
    print("   • Full workflow: operation='upload_and_serve'") 