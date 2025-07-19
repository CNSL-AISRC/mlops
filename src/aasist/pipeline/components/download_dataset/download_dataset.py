from kfp import dsl
from dotenv import load_dotenv
import os

load_dotenv()

download_dataset_component = {
    "base_image": "python:3.11",  # Changed from alpine to standard python
    "target_image": f"{os.getenv('PRIVATE_DOCKER_REGISTRY')}/aasist-project/download-dataset:{os.getenv('DOWNLOAD_DATASET_VERSION')}",
    "packages_to_install": [
        'requests==2.32.4',
        'python-dotenv==1.1.1',
        'tqdm==4.67.1'
    ]
}

@dsl.component(**download_dataset_component)
def download_dataset(
    dataset_url: str
) -> str:
    """
    Download dataset and save to PVC storage.
    
    Args:
        dataset_url: URL to download the dataset from
        
    Returns:
        Path to the downloaded file in PVC
    """
    import requests
    import os
    from tqdm import tqdm
    from urllib.parse import urlparse
    
    # PVC mount path
    pvc_path = "/data"
    
    # Extract filename from URL
    parsed_url = urlparse(dataset_url)
    filename = os.path.basename(parsed_url.path)
    dst_path = os.path.join(pvc_path, filename)
    # Check if filename is exists in the pvc path
    if os.path.exists(os.path.join(pvc_path, filename)):
        print(f"Dataset already exists in {pvc_path}")
        return dst_path

    print(f"Downloading dataset from: {dataset_url}")
    print(f"Saving to: {dst_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(pvc_path, exist_ok=True)
    
    # Download with progress bar
    response = requests.get(dataset_url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dst_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f"Dataset downloaded successfully to {dst_path}")
    return dst_path 