from kfp import dsl
from dotenv import load_dotenv
import os

load_dotenv()

download_dataset_component = {
    "base_image": "python:3.11",
    "target_image": f"{os.getenv('PRIVATE_DOCKER_REGISTRY')}/aasist-project/download-dataset:{os.getenv('DOWNLOAD_DATASET_VERSION')}",
    "packages_to_install": [
        'requests==2.32.4',
        'python-dotenv==1.1.1'
    ]
}

@dsl.component(**download_dataset_component)
def download_dataset(dataset_url: str = "http://10.5.110.131:8080/test.zip") -> str:
    """
    Download a dataset file from a URL.
    Returns the path to the downloaded file.
    """
    import requests
    import os
    
    # Use a fixed shared location that both components can access
    download_path = "/tmp/shared/downloaded_dataset.zip"
    os.makedirs("/tmp/shared", exist_ok=True)
    
    print(f"Downloading dataset from {dataset_url}")
    print(f"Download path: {download_path}")

    response = requests.get(dataset_url, stream=True)
    response.raise_for_status()

    with open(download_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    file_size = os.path.getsize(download_path)
    print(f"Dataset downloaded successfully, size: {file_size} bytes")
    
    # Return the path as a string
    return download_path