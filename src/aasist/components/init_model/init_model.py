from kfp import dsl
from dotenv import load_dotenv
import os
from kfp.dsl import Output
from kfp.dsl import Model
load_dotenv()

init_model_component = {
    "base_image": "python:3.11",
    "target_image": f"{os.getenv('PRIVATE_DOCKER_REGISTRY')}/aasist-project/init-model:{os.getenv('INIT_MODEL_VERSION')}",
    "packages_to_install": [
        'torch==2.7.1',
        'numpy==2.3.1',
        'dotenv'
    ]
}

@dsl.component(
    **init_model_component
)
def init_model(
    config_str: str,
    device: str = "cuda",
    model_artifact: Output[Model] = None
):
    """
    Initialize model architecture and optionally load pretrained weights.
    
    Args:
        config_str: JSON string containing model configuration
        device: Device to load model on ("cuda" or "cpu")
        
    Returns:
        Path to the saved initialized model
    """
    import json
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Union
    import random
    import numpy as np
    from pathlib import Path
    from typing import Dict
    from importlib import import_module
    #from utils import get_model
    
    def get_model(model_config: Dict, device: torch.device):
        """Define DNN model architecture"""
        module = import_module("models.{}".format(model_config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(model_config).to(device)
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        print("no. model params:{}".format(nb_params))
        return model
    
    # Parse configuration
    config = json.loads(config_str)
    model_config = config["model_config"]
    pretrained_model_path = config.get("model_path", "")
    print(f"Initializing model: {model_config['architecture']}")
    print(f"Device: {device}")
    
    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    model = get_model(model_config, device)
    # Load pretrained weights if provided
    if pretrained_model_path and pretrained_model_path.strip():
        print(f"Loading pretrained weights from {pretrained_model_path}")
        try:
            state_dict = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Pretrained weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights...")
    
    with open(model_artifact.path, "wb") as f:
        torch.save(model.state_dict(), f)