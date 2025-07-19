#!/usr/bin/env python3

import os
import logging
import kserve
from kserve import ModelServer
import mlflow
import torch
from typing import Dict
import numpy as np

class ModelWrapper(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        KSERVE_LOGGER_NAME = 'kserve'
        self.logger = logging.getLogger(KSERVE_LOGGER_NAME)
        self.name = name
        self.ready = False
        self.model_uri = os.getenv("MODEL_URI")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initializing ModelWrapper with model_uri: {self.model_uri}")
        
    def load(self):
        """Load the MLflow model"""
        try:
            self.logger.info(f"Loading model from: {self.model_uri}")
            self.model = mlflow.pytorch.load_model(self.model_uri)
            self.ready = True
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e
    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x
    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        """
        Perform inference on the input data
        Expected input format: {"inputs": [[...], [...], ...]}
        Returns: {"predictions": [[...], [...], ...]}
        """
        try:
            self.logger.info(f"Received prediction request: {request}")
            
            # Handle both "inputs" and "instances" format for compatibility
            if "inputs" in request:
                inputs_data = request["inputs"]
            elif "instances" in request:
                inputs_data = request["instances"]
            else:
                raise ValueError("Request must contain either 'inputs' or 'instances' field")
            
            # Convert to tensor
            inputs = torch.tensor(inputs_data, dtype=torch.float32)
            # pad the input to 64600
            inputs = self.pad(inputs)
            # convert to tensor
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            # reshape reverse of the input
            inputs = inputs.reshape(1, -1)
            self.logger.info(f"Input tensor shape: {inputs.shape}")
            
            # Make prediction
            with torch.no_grad():
                _, y_pred = self.model(inputs)
                #self.logger.info(f"Raw prediction shape: {y_pred.shape}")
                
                # Apply softmax
                y_pred = torch.softmax(y_pred, dim=1)
                #self.logger.info(f"Prediction after softmax: {y_pred}")
                
                # Convert to list for JSON serialization
                predictions = y_pred.tolist()
            
            return {"predictions": predictions}
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise e

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get model name from environment
    model_name = os.getenv("MODEL_NAME", "pytorch-model")
    logger.info(f"Starting model server for: {model_name}")
    
    # Create model wrapper
    model = ModelWrapper(model_name)
    model.load()
    
    # Start the model server
    logger.info("Starting KServe ModelServer...")
    ModelServer().start([model]) 