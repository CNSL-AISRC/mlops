from kfp import dsl
from dotenv import load_dotenv
import os

load_dotenv()

extract_dataset_component = {
    "base_image": "python:3.11",
    "target_image": f"{os.getenv('PRIVATE_DOCKER_REGISTRY')}/aasist-project/extract-dataset:{os.getenv('EXTRACT_DATASET_VERSION')}",
    "packages_to_install": [
        'python-dotenv==1.1.1'
    ]
}

@dsl.component(**extract_dataset_component)
def extract_dataset(zip_file_path: str) -> str:
    """
    Extract a dataset archive and return the extraction directory path.
    """
    import zipfile
    import os

    print(f"Extracting dataset from {zip_file_path}")
    
    # Extract to a shared location
    extraction_dir = "/tmp/shared/dataset_extracted"
    os.makedirs(extraction_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
    
    print(f"Dataset extracted successfully to {extraction_dir}")
    
    # Return the extraction directory path
    return extraction_dir 