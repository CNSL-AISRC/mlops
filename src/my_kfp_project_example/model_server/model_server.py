#!/usr/bin/env python3

import os
import logging
import kserve
from kserve import ModelServer
import mlflow
import torch
from typing import Dict, Optional
import numpy as np
import gc

class ModelWrapper(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        KSERVE_LOGGER_NAME = 'kserve'
        self.logger = logging.getLogger(KSERVE_LOGGER_NAME)
        self.name = name
        self.ready = False
        self.model_uri = os.getenv("MODEL_URI")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.max_len = 64600  # Cache the max length as instance variable
        self.logger.info(f"Initializing ModelWrapper with model_uri: {self.model_uri}")
        
        # Pre-allocate tensor for padding to reduce memory allocations
        if self.device.type == "cuda":
            # Warm up CUDA context
            torch.cuda.empty_cache()
        
    def load(self):
        """Load the MLflow model"""
        try:
            self.logger.info(f"Loading model from: {self.model_uri}")
            self.model = mlflow.pytorch.load_model(self.model_uri, map_location=self.device)
            
            # Set model to evaluation mode to disable dropout/batch norm training behavior
            self.model.eval()
            
            # Move model to device if not already there
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            self.ready = True
            self.logger.info("Model loaded successfully")
            
            # Clear any temporary memory from model loading
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e
            
    def pad(self, x: np.ndarray, max_len: int = None) -> np.ndarray:
        """Optimized padding function with memory efficiency - supports batch padding"""
        if max_len is None:
            max_len = self.max_len
        
        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample
            x_len = x.shape[0]
            if x_len >= max_len:
                return x[:max_len]
            
            if x_len == 0:
                return np.zeros(max_len, dtype=x.dtype)
            
            # Calculate repetitions needed
            num_repeats = max_len // x_len
            remainder = max_len % x_len
            
            if remainder == 0:
                return np.tile(x, num_repeats)
            else:
                padded_x = np.empty(max_len, dtype=x.dtype)
                if num_repeats > 0:
                    full_part = np.tile(x, num_repeats)
                    padded_x[:len(full_part)] = full_part
                if remainder > 0:
                    start_idx = num_repeats * x_len
                    padded_x[start_idx:start_idx + remainder] = x[:remainder]
                return padded_x
                
        elif x.ndim == 2:
            # Batch of samples - pad each sample in the batch
            batch_size = x.shape[0]
            padded_batch = np.empty((batch_size, max_len), dtype=x.dtype)
            
            for i in range(batch_size):
                sample = x[i]
                x_len = sample.shape[0]
                
                if x_len >= max_len:
                    padded_batch[i] = sample[:max_len]
                elif x_len == 0:
                    padded_batch[i] = np.zeros(max_len, dtype=x.dtype)
                else:
                    # Calculate repetitions needed
                    num_repeats = max_len // x_len
                    remainder = max_len % x_len
                    
                    if remainder == 0:
                        padded_batch[i] = np.tile(sample, num_repeats)
                    else:
                        if num_repeats > 0:
                            full_part = np.tile(sample, num_repeats)
                            padded_batch[i][:len(full_part)] = full_part
                        if remainder > 0:
                            start_idx = num_repeats * x_len
                            padded_batch[i][start_idx:start_idx + remainder] = sample[:remainder]
            
            return padded_batch
        else:
            raise ValueError(f"Input must be 1D or 2D, got {x.ndim}D")
    
    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Perform inference on the input data with efficient batch processing
        Expected input format: {"inputs": [[...], [...], ...]}
        Returns: {"predictions": [[...], [...], ...]}
        """
        if not self.ready or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        try:
            # Log request size instead of full content to reduce memory usage
            if "inputs" in request:
                inputs_data = request["inputs"]
                self.logger.info(f"Received prediction request with {len(inputs_data)} input(s)")
            elif "instances" in request:
                inputs_data = request["instances"]
                self.logger.info(f"Received prediction request with {len(inputs_data)} instance(s)")
            else:
                raise ValueError("Request must contain either 'inputs' or 'instances' field")
            
            # Convert to numpy first for more efficient processing
            if isinstance(inputs_data, list):
                inputs_np = np.array(inputs_data, dtype=np.float32)
            else:
                inputs_np = np.asarray(inputs_data, dtype=np.float32)
            
            # Ensure we have a 2D array (batch_size, features)
            if inputs_np.ndim == 1:
                # Single input - add batch dimension
                inputs_np = inputs_np.reshape(1, -1)
            elif inputs_np.ndim == 2:
                # Already batched - this is what we want
                pass
            else:
                raise ValueError(f"Input must be 1D or 2D, got {inputs_np.ndim}D")
            
            # Pad the entire batch at once - much more efficient!
            padded_inputs = self.pad(inputs_np)
            self.logger.info(f"Batch input shape after padding: {padded_inputs.shape}")
            
            # Convert entire batch to tensor directly on target device
            input_tensor = torch.from_numpy(padded_inputs).to(
                device=self.device, 
                dtype=torch.float32, 
                non_blocking=True if self.device.type == "cuda" else False
            )
            
            # Make prediction on entire batch - much more efficient!
            with torch.no_grad():
                # Set model to eval mode (in case it was changed elsewhere)
                self.model.eval()
                
                _, y_pred = self.model(input_tensor)
                
                # Apply softmax
                y_pred = torch.softmax(y_pred, dim=1)
                
                # Convert to CPU and then to list to free GPU memory immediately
                predictions = y_pred.cpu().tolist()
            
            # Cleanup tensors
            del input_tensor, y_pred
            
            # Clear GPU cache after processing
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            return {"predictions": predictions}
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Cleanup on error
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise e
    
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'device') and self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore cleanup errors

if __name__ == "__main__":
    # Setup logging with more efficient formatting
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Get model name from environment
    model_name = os.getenv("MODEL_NAME", "pytorch-model")
    logger.info(f"Starting model server for: {model_name}")
    
    try:
        # Create model wrapper
        model = ModelWrapper(model_name)
        model.load()
        
        # Start the model server
        logger.info("Starting KServe ModelServer...")
        ModelServer().start([model])
    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        raise
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 