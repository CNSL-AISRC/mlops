# AASIST Demo Pipeline

A clean, simplified pipeline for demonstrating AASIST anti-spoofing functionality using Kubeflow Pipelines with PVC data sharing.

## Structure

```
pipeline/
├── components/           # Pipeline components
│   ├── download_dataset.py  # Downloads dataset to PVC
│   └── extract_dataset.py   # Extracts dataset from PVC
├── pipeline.py          # Main pipeline entry point
└── README.md           # This file
```

## Features

- **PVC Data Sharing**: Uses Persistent Volume Claims for efficient data sharing between components
- **Simple Client Setup**: Based on the working test.py client configuration
- **Clean Architecture**: Minimal, focused code without unnecessary complexity
- **Demo-focused**: Designed for quick testing and demonstration

## Prerequisites

1. **PVC Setup**: Create a PVC named `pipeline-data-pvc` in your cluster:
   ```yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: pipeline-data-pvc
     namespace: admin
   spec:
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 10Gi
   ```

2. **Kubeconfig**: Ensure your `~/.kube/config` is properly configured for cluster access

## Usage

### Run the Demo Pipeline
```bash
cd pipeline
python pipeline.py
```

### Compile Only (without running)
```bash
python pipeline.py --compile-only
```

### Custom Host/Namespace
```bash
python pipeline.py --host http://your-kfp-host:port --namespace your-namespace
```

## Pipeline Steps

1. **Download Dataset**: Downloads the dataset from the specified URL to PVC storage
2. **Extract Dataset**: Extracts the downloaded zip file within the PVC

Both steps share data through the PVC, making the pipeline efficient and following Kubeflow best practices.

## Configuration

Default settings:
- **KFP Host**: `http://10.5.110.131:31047`
- **Namespace**: `admin`
- **Dataset URL**: `http://10.5.110.131:8080/test.zip`
- **PVC Name**: `pipeline-data-pvc`

All settings can be customized through command-line arguments or by modifying the pipeline code. 