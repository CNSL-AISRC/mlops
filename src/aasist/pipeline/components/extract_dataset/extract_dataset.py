from kfp import dsl
from dotenv import load_dotenv
import os

load_dotenv()

extract_dataset_component = {
    "base_image": "python:3.11",  # Changed from alpine to standard python
    "target_image": f"{os.getenv('PRIVATE_DOCKER_REGISTRY')}/aasist-project/extract-dataset:{os.getenv('EXTRACT_DATASET_VERSION')}",
    "packages_to_install": [
        'python-dotenv==1.1.1'
    ]
}

@dsl.component(**extract_dataset_component)
def extract_dataset(
    dataset_path: str
) -> str:
    """
    Extract dataset from zip file stored in PVC.
    
    Args:
        zip_file_path: Path to the zip file in PVC
        pvc_name: Name of the PVC containing the data
        
    Returns:
        Path to the extracted dataset directory in PVC
    """
    import zipfile
    import os
    from pathlib import Path
    print(f"Dataset path: {dataset_path}")
    # PVC mount path
    pvc_path = "/data"
    zip_file_path = "test.zip"
    zip_file_path = os.path.join(pvc_path, zip_file_path)
    # Check if extraction_dir is exists in the pvc path
    extraction_dir = os.path.join(pvc_path, "extracted_dataset")
        # Create extraction directory
    os.makedirs(extraction_dir, exist_ok=True)
    # if extraction_dir is empty 
    if not os.listdir(extraction_dir):
        print(f"Extraction directory is empty")
        print(f"Extracting dataset from: {zip_file_path}")
        print(f"Extraction directory: {extraction_dir}")
        
        # Extract the zip file
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
            
            print("Dataset extracted successfully!")
            
            # List extracted contents
            print("Extracted contents:")
            for root, dirs, files in os.walk(extraction_dir):
                level = root.replace(extraction_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files only
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
            
            return extraction_dir
            
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            raise e 
    else:
        print(f"Extraction directory is not empty")
        return extraction_dir
    

    
    