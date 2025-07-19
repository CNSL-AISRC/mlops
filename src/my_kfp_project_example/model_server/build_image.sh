#!/bin/bash

# Load environment variables
set -a
source ../.env
set +a

# Build and push the custom model server image
REGISTRY=${PRIVATE_DOCKER_REGISTRY}
IMAGE_NAME="aasist-project/model-server"
VERSION=${SERVING_MODEL_VERSION}
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo "Building custom model server image: ${FULL_IMAGE_NAME}"

# Build the image
docker build -t ${FULL_IMAGE_NAME} .

# Push the image
echo "Pushing image to registry..."
docker push ${FULL_IMAGE_NAME}

echo "✅ Custom model server image built and pushed successfully: ${FULL_IMAGE_NAME}" 