from kfp import dsl

@dsl.component(
    packages_to_install=[
        'requests==2.32.4',
        'tqdm==4.67.1'
    ]
)
def download_data(
    dataset_url: str,
    pvc_path: str
    ) -> str:
    # print(f"Downloading data from {dataset_url}")
    # return dataset_url
    import requests
    import os
    from tqdm import tqdm
    from urllib.parse import urlparse
    
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
