from kfp import dsl
import os
from dotenv import load_dotenv

load_dotenv()


@dsl.component(
    base_image='python:3.11',
    target_image=f'{os.getenv("PRIVATE_DOCKER_REGISTRY")}/aasist-project/train-model:{os.getenv("TRAIN_MODEL_VERSION")}',
    packages_to_install=[
        'torch==2.7.1',
        'torchcontrib==0.0.2',
        'numpy==2.3.1',
        'soundfile==0.13.1',
        'tensorboard==2.19.0',
        'tqdm==4.67.1',
        'dotenv'
    ]
)
def train_model(processed_data: str, config: str) -> str:
    import argparse
    import json
    import os
    import random
    import sys
    import warnings
    from importlib import import_module
    from pathlib import Path
    from shutil import copy
    from typing import Dict, List, Union
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchcontrib.optim import SWA
    from tqdm import tqdm

    from data_utils import (Dataset_ASVspoof2019_train,
                            Dataset_ASVspoof2019_devNeval, genSpoof_list)
    from evaluation import calculate_tDCF_EER
    from utils import create_optimizer, seed_worker, set_seed, str_to_bool

    # load experiment configurations
    with open(config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    set_seed(seed, config)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    model = get_model(model_config, device)

    model = f"model_trained_on({processed_data})"
    print("Trained model:", model)
    return model
