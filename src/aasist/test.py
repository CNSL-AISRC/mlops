#!/usr/bin/env python3
"""
Test script for AASIST Serving Service
Tests the deployed serving endpoints after run_serving_only.py completes successfully
"""
import requests
import numpy as np
import json
import time
import sys
from typing import Dict, Any, List

# Service configuration
SERVICE_BASE_URLS = [
    # Real AASIST KServe URLs (from kubeflow_pipeline_real.py)
    "http://aasist-real.admin.svc.cluster.local/v1/models/aasist-real-model:predict",  # Real KServe v1 endpoint
    "http://aasist-real-predictor-default.admin.svc.cluster.local",  # Real KServe predictor
    "http://aasist-real.admin.svc.cluster.local",  # Real base KServe URL
    # Minimal AASIST KServe URLs (from kubeflow_pipeline_minimal.py)
    "http://aasist-minimal.admin.svc.cluster.local/v1/models/aasist-model:predict",  # Minimal KServe v1 endpoint
    "http://aasist-minimal-predictor-default.admin.svc.cluster.local",  # Minimal KServe predictor
    "http://aasist-minimal.admin.svc.cluster.local",  # Minimal base KServe URL
    # Legacy URLs from other pipelines
    "http://aasist-serving-direct.admin.svc.cluster.local:5000",  # Direct access
    "http://aasist-serving.admin.svc.cluster.local:5000",  # Main service
    "http://localhost:8080",  # Local KServe default port
    "http://127.0.0.1:8080",  # Alternative local
]

def generate_mock_audio_data(duration_seconds: float = 1.0, sample_rate: int = 16000) -> List[float]:
    """Generate mock audio data for testing"""
    num_samples = int(duration_seconds * sample_rate)
    # Generate realistic audio-like data (sine wave with noise)
    t = np.linspace(0, duration_seconds, num_samples)
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)  # Sine wave
    noise = 0.1 * np.random.randn(num_samples)  # Add some noise
    audio_with_noise = audio + noise
    return audio_with_noise.tolist()

def test_health_endpoint(base_url: str) -> Dict[str, Any]:
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Health check passed")
            print(f"  📊 Status: {data.get('status', 'unknown')}")
            print(f"  🏷️  Service: {data.get('service', 'unknown')}")
            print(f"  ⚙️  Config: {data.get('config', 'unknown')}")
            return {"status": "success", "data": data, "response_time": response.elapsed.total_seconds()}
        else:
            print(f"  ❌ Health check failed: HTTP {response.status_code}")
            return {"status": "error", "error": f"HTTP {response.status_code}", "response": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Health check failed: {e}")
        return {"status": "error", "error": str(e)}

def test_info_endpoint(base_url: str) -> Dict[str, Any]:
    """Test the model info endpoint"""
    print("🔍 Testing info endpoint...")
    try:
        response = requests.get(f"{base_url}/info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Info endpoint working")
            print(f"  🤖 Service: {data.get('service_name', 'unknown')}")
            print(f"  📁 Model: {data.get('model_path', 'unknown')}")
            print(f"  ⚙️  Config: {data.get('config_name', 'unknown')}")
            print(f"  🔗 Endpoints: {len(data.get('endpoints', {}))}")
            return {"status": "success", "data": data, "response_time": response.elapsed.total_seconds()}
        else:
            print(f"  ❌ Info endpoint failed: HTTP {response.status_code}")
            return {"status": "error", "error": f"HTTP {response.status_code}", "response": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Info endpoint failed: {e}")
        return {"status": "error", "error": str(e)}

def test_predict_endpoint(base_url: str) -> Dict[str, Any]:
    """Test the prediction endpoint with mock audio"""
    print("🔍 Testing prediction endpoint...")
    try:
        # Generate mock audio data
        audio_data = generate_mock_audio_data(duration_seconds=2.0, sample_rate=16000)
        
        payload = {
            "audio_data": audio_data,
            "sample_rate": 16000
        }
        
        print(f"  📊 Sending audio: {len(audio_data)} samples, {len(audio_data)/16000:.1f}s duration")
        
        response = requests.post(
            f"{base_url}/predict", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Prediction successful")
            print(f"  🎯 Prediction: {data.get('prediction', 'unknown')}")
            print(f"  📈 Score: {data.get('score', 'unknown')}")
            print(f"  🎪 Confidence: {data.get('confidence', 'unknown')}")
            print(f"  ⏱️  Processing time: {data.get('processing_info', {}).get('processing_time_ms', 'unknown')}ms")
            return {"status": "success", "data": data, "response_time": response.elapsed.total_seconds()}
        else:
            print(f"  ❌ Prediction failed: HTTP {response.status_code}")
            print(f"  📄 Response: {response.text}")
            return {"status": "error", "error": f"HTTP {response.status_code}", "response": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Prediction failed: {e}")
        return {"status": "error", "error": str(e)}

def test_batch_predict_endpoint(base_url: str) -> Dict[str, Any]:
    """Test batch prediction if available"""
    print("🔍 Testing batch prediction endpoint...")
    try:
        # Generate multiple mock audio samples
        instances = [
            {"audio_data": generate_mock_audio_data(1.0, 16000), "sample_rate": 16000},
            {"audio_data": generate_mock_audio_data(1.5, 16000), "sample_rate": 16000},
            {"audio_data": generate_mock_audio_data(2.0, 16000), "sample_rate": 16000}
        ]
        
        payload = {"instances": instances}
        
        print(f"  📊 Sending {len(instances)} audio samples for batch prediction")
        
        response = requests.post(
            f"{base_url}/batch_predict", 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"  ✅ Batch prediction successful")
            print(f"  🎯 Predictions: {len(predictions)} results")
            for i, pred in enumerate(predictions[:3]):  # Show first 3
                print(f"    {i+1}. {pred.get('prediction', 'unknown')} (confidence: {pred.get('confidence', 'unknown')})")
            return {"status": "success", "data": data, "response_time": response.elapsed.total_seconds()}
        else:
            print(f"  ⚠️  Batch prediction not available or failed: HTTP {response.status_code}")
            return {"status": "not_available", "error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️  Batch prediction not available: {e}")
        return {"status": "not_available", "error": str(e)}

def find_service_url() -> str:
    """Try to find the working service URL"""
    print("🔍 Searching for active service...")
    
    for url in SERVICE_BASE_URLS:
        print(f"  🔗 Trying: {url}")
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ Found active service at: {url}")
                return url
        except:
            print(f"  ❌ No response from: {url}")
            continue
    
    return None

def run_comprehensive_test(base_url: str) -> Dict[str, Any]:
    """Run all tests and return comprehensive results"""
    print(f"\n🧪 Running comprehensive test suite on: {base_url}")
    print("=" * 60)
    
    results = {
        "service_url": base_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }
    
    # Test 1: Health check
    results["tests"]["health"] = test_health_endpoint(base_url)
    print()
    
    # Test 2: Model info
    results["tests"]["info"] = test_info_endpoint(base_url)
    print()
    
    # Test 3: Single prediction
    results["tests"]["predict"] = test_predict_endpoint(base_url)
    print()
    
    # Test 4: Batch prediction (optional)
    results["tests"]["batch_predict"] = test_batch_predict_endpoint(base_url)
    print()
    
    return results

def print_summary(results: Dict[str, Any]):
    """Print test summary"""
    print("📋 TEST SUMMARY")
    print("=" * 30)
    
    total_tests = len(results["tests"])
    passed_tests = 0
    
    for test_name, test_result in results["tests"].items():
        status = test_result.get("status", "unknown")
        if status == "success":
            icon = "✅"
            passed_tests += 1
        elif status == "not_available":
            icon = "⚠️"
        else:
            icon = "❌"
        
        response_time = test_result.get("response_time", 0)
        print(f"  {icon} {test_name}: {status} ({response_time*1000:.0f}ms)")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} core tests passed")
    
    if passed_tests >= 3:  # Health, Info, Predict
        print("🎉 Service is working correctly!")
        return True
    else:
        print("❌ Service has issues - check the logs above")
        return False

def main():
    """Main test function"""
    print("🤖 AASIST Serving Service Test")
    print("=" * 40)
    print("Testing endpoints after run_serving_only.py pipeline completion...")
    print()
    
    # Find the service
    service_url = find_service_url()
    
    if not service_url:
        print("❌ Could not find active AASIST serving service!")
        print("\n💡 Troubleshooting:")
        print("1. Make sure run_serving_only.py completed successfully")
        print("2. Check if the service is running in your cluster:")
        print("   kubectl get pods -n kubeflow")
        print("3. Check service endpoints:")
        print("   kubectl get services -n kubeflow")
        print("4. Try port forwarding if testing locally:")
        print("   kubectl port-forward svc/aasist-serving 5000:5000 -n kubeflow")
        sys.exit(1)
    
    # Run comprehensive tests
    results = run_comprehensive_test(service_url)
    
    # Print summary
    success = print_summary(results)
    
    # Save results to file
    results_file = "test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Full results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 