"""
Custom KServe Predictor for AASIST Models
Handles audio anti-spoofing inference with AASIST models
"""
import os
import json
import torch
import numpy as np
import soundfile as sf
from typing import Dict, List, Union
from io import BytesIO
import base64
import traceback
import sys

# KServe imports
try:
    from kserve import Model, ModelServer
    from kserve.utils.utils import generate_uuid
except ImportError:
    print("KServe not available, using mock classes")
    class Model:
        def __init__(self, name: str):
            self.name = name
        def load(self): pass
        def predict(self, payload: Dict, headers: Dict = None): pass
    class ModelServer:
        @staticmethod
        def start(models): pass

class AASISTPredictor(Model):
    """
    Custom predictor for AASIST anti-spoofing models
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.device = None
        self.model_config = None
        self.ready = False
        
    def load(self) -> bool:
        """Load the AASIST model"""
        try:
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # Load model configuration
            config_path = os.environ.get("MODEL_CONFIG_PATH", "/mnt/models/model_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)
            else:
                # Default AASIST configuration
                self.model_config = {
                    "architecture": "AASIST",
                    "nb_samp": 64600,
                    "first_conv": 128,
                    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                    "gat_dims": [64, 32],
                    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
                    "temperatures": [2.0, 2.0, 100.0, 100.0]
                }
            
            # Load model architecture
            model_path = os.environ.get("MODEL_PATH", "/mnt/models")
            sys.path.append(model_path)
            
            # Import AASIST model (assuming it's available in the container)
            try:
                from models.AASIST import Model as AASISTModel
                self.model = AASISTModel(self.model_config, self.device)
            except ImportError:
                print("AASIST model not found, using mock model")
                self.model = MockAASISTModel(self.model_config, self.device)
            
            # Load model weights
            weights_path = os.environ.get("MODEL_WEIGHTS_PATH", "/mnt/models/best.pth")
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model weights from {weights_path}")
            else:
                print("Warning: No model weights found, using randomly initialized model")
            
            self.model.to(self.device)
            self.model.eval()
            
            self.ready = True
            print("AASIST model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, payload: Dict, headers: Dict = None) -> Dict:
        """
        Predict spoofing probability for audio input
        
        Expected payload formats:
        1. Base64 encoded audio: {"audio_b64": "base64_string", "sample_rate": 16000}
        2. Audio file URL: {"audio_url": "http://...", "sample_rate": 16000}
        3. Raw audio array: {"audio_data": [...], "sample_rate": 16000}
        """
        if not self.ready:
            return {"error": "Model not loaded"}
        
        try:
            # Parse input
            audio_data, sample_rate = self._parse_audio_input(payload)
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_data, sample_rate)
            
            # Run inference
            with torch.no_grad():
                # Convert to tensor and add batch dimension
                audio_tensor = torch.FloatTensor(processed_audio).unsqueeze(0).to(self.device)
                
                # Get model prediction
                output = self.model(audio_tensor)
                
                # Extract prediction scores
                if isinstance(output, tuple):
                    # If model returns (embedding, prediction)
                    prediction = output[1]
                else:
                    prediction = output
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(prediction, dim=1)
                bonafide_prob = probabilities[0, 1].cpu().item()  # Probability of bonafide
                spoof_prob = probabilities[0, 0].cpu().item()     # Probability of spoof
                
                # Calculate confidence score
                confidence = max(bonafide_prob, spoof_prob)
                
                # Determine prediction
                is_bonafide = bonafide_prob > spoof_prob
                
            result = {
                "prediction": "bonafide" if is_bonafide else "spoof",
                "confidence": confidence,
                "probabilities": {
                    "bonafide": bonafide_prob,
                    "spoof": spoof_prob
                },
                "model_info": {
                    "name": self.name,
                    "architecture": self.model_config.get("architecture", "AASIST"),
                    "device": str(self.device)
                }
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {"error": error_msg}
    
    def _parse_audio_input(self, payload: Dict) -> tuple:
        """Parse different audio input formats"""
        sample_rate = payload.get("sample_rate", 16000)
        
        if "audio_b64" in payload:
            # Base64 encoded audio
            audio_bytes = base64.b64decode(payload["audio_b64"])
            audio_data, sr = sf.read(BytesIO(audio_bytes))
            if sr != sample_rate:
                print(f"Warning: Expected sample rate {sample_rate}, got {sr}")
            return audio_data, sr
            
        elif "audio_url" in payload:
            # Audio file URL (for demo purposes, not recommended in production)
            import requests
            response = requests.get(payload["audio_url"])
            audio_data, sr = sf.read(BytesIO(response.content))
            return audio_data, sr
            
        elif "audio_data" in payload:
            # Raw audio array
            audio_data = np.array(payload["audio_data"])
            return audio_data, sample_rate
            
        else:
            raise ValueError("No valid audio input found in payload")
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio for AASIST model"""
        # Ensure mono audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if necessary (simplified - in production use proper resampling)
        target_length = self.model_config.get("nb_samp", 64600)
        
        if len(audio_data) > target_length:
            # Truncate
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            # Pad with repetition (as in original AASIST preprocessing)
            num_repeats = int(target_length / len(audio_data)) + 1
            audio_data = np.tile(audio_data, num_repeats)[:target_length]
        
        return audio_data.astype(np.float32)


class MockAASISTModel:
    """Mock AASIST model for testing when real model is not available"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
    def to(self, device):
        return self
        
    def eval(self):
        return self
        
    def load_state_dict(self, state_dict):
        pass
        
    def __call__(self, x):
        # Return mock prediction
        batch_size = x.size(0)
        # Random predictions for demo
        logits = torch.randn(batch_size, 2).to(self.device)
        return logits


def create_model_server():
    """Create and configure the model server"""
    model = AASISTPredictor("aasist-predictor")
    model.load()
    
    # Start the model server
    ModelServer().start([model])


if __name__ == "__main__":
    create_model_server() 