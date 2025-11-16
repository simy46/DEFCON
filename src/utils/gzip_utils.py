import os
import gzip
import shutil

def gunzip_file(gz_path: str, output_path: str) -> None:
    """Extract a .gz file to a target path."""
    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"GZIP file not found: {gz_path}")

    print(f"Decompressing {gz_path} → {output_path} ...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Decompression completed.")