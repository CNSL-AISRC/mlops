# Model Server

This directory contains the custom KServe model server implementation for serving PyTorch models via MLflow.

## Structure

- `model_server.py` - Main server script with ModelWrapper implementation
- `Dockerfile` - Docker image definition for the model server
- `build_image.sh` - Script to build and push the model server image

## Building the Model Server Image

```bash
cd src/my_kfp_project_example/model_server
chmod +x build_image.sh
./build_image.sh
```

This will build and push the image: `{PRIVATE_DOCKER_REGISTRY}/aasist-project/model-server:{SERVING_MODEL_VERSION}`

## Purpose

This model server image is used by the KServe InferenceService created by the `serving_comp` pipeline component. It contains:

- Custom ModelWrapper class that loads MLflow PyTorch models
- KServe-compatible inference interface
- Support for GPU acceleration
- Environment-based configuration

## Separation from Pipeline Component

This directory is separate from the pipeline component (`components/serving_comp/`) to avoid confusion during pipeline compilation. The component uses a lightweight Python image for orchestration, while this creates the heavy ML serving image. 