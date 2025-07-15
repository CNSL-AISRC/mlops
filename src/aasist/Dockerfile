FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=2.3.1 \
    soundfile>=0.13.1 \
    tensorboard>=2.19.0 \
    torchcontrib>=0.0.2 \
    tqdm>=4.67.1 \
    mlflow==2.15.1 \
    boto3==1.34.162 \
    kserve==0.13.1 \
    kubernetes==26.1.0 \
    tenacity==9.0.0 \
    requests \
    zipfile36

# Copy AASIST source code
COPY . /workspace/aasist/

# Set working directory to AASIST
WORKDIR /workspace/aasist

# Make scripts executable
RUN chmod +x *.py

# Set Python path to include the workspace
ENV PYTHONPATH="/workspace/aasist:$PYTHONPATH"

# Default command
CMD ["python", "distributed_main.py", "--help"] 