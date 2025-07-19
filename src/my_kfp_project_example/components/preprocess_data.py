from kfp import dsl

@dsl.component
def preprocess_data(zip_file_path: str) -> str:
    # cleaned = raw_data.upper()  # Dummy processing
    # print("Cleaned:", cleaned)
    import zipfile
    import os
    from pathlib import Path

    # Check if extraction_dir is exists in the pvc path
    extraction_dir = os.path.join(os.path.dirname(zip_file_path), "src")
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
