# app/utils/document_processor.py - Simplified version
import os
import glob
from typing import List

def list_documents(directory: str) -> List[str]:
    """List all documents in the given directory."""
    all_files = []
    
    # Check all supported file types
    for ext in ["*.txt", "*.pdf", "*.docx"]:
        all_files.extend(glob.glob(os.path.join(directory, ext)))
    
    # Extract just the filenames
    filenames = [os.path.basename(file) for file in all_files]
    
    return filenames

def save_uploaded_file(file_content: bytes, filename: str, directory: str) -> str:
    """Save uploaded file to the documents directory."""
    os.makedirs(directory, exist_ok=True)
    
    file_path = os.path.join(directory, filename)
    
    # Write file to disk
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return file_path

def delete_document(filename: str, directory: str) -> bool:
    """Delete a document from the documents directory."""
    file_path = os.path.join(directory, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    
    return False