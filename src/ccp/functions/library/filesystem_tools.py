"""
Filesystem management tools for CCP.
Provides safe file operations with proper error handling.
"""
import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional
from src.ccp.functions.utils import expose_as_ccp_tool


@expose_as_ccp_tool
def read_file(file_path: str) -> str:
    """
    Read contents of a file.
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        File contents as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


@expose_as_ccp_tool
def write_file(file_path: str, content: str, mode: str = "w") -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        mode: Write mode ('w' for overwrite, 'a' for append)
    
    Returns:
        Success message or error
    """
    try:
        # Create parent directories if they don't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@expose_as_ccp_tool
def list_directory(directory_path: str, pattern: str = "*") -> str:
    """
    List files and directories in a path.
    
    Args:
        directory_path: Path to directory
        pattern: Glob pattern for filtering (e.g., "*.py", "*.txt")
    
    Returns:
        JSON string of directory contents
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return json.dumps({"error": f"Directory {directory_path} does not exist"})
        
        items = []
        for item in path.glob(pattern):
            items.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None
            })
        
        return json.dumps({"directory": directory_path, "items": items}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@expose_as_ccp_tool
def create_directory(directory_path: str) -> str:
    """
    Create a directory (including parent directories).
    
    Args:
        directory_path: Path to directory to create
    
    Returns:
        Success message or error
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory: {directory_path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"


@expose_as_ccp_tool
def delete_file(file_path: str) -> str:
    """
    Delete a file.
    
    Args:
        file_path: Path to file to delete
    
    Returns:
        Success message or error
    """
    try:
        Path(file_path).unlink()
        return f"Successfully deleted: {file_path}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"


@expose_as_ccp_tool
def copy_file(source: str, destination: str) -> str:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
    
    Returns:
        Success message or error
    """
    try:
        # Create destination directory if needed
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return f"Successfully copied {source} to {destination}"
    except Exception as e:
        return f"Error copying file: {str(e)}"


@expose_as_ccp_tool
def move_file(source: str, destination: str) -> str:
    """
    Move a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
    
    Returns:
        Success message or error
    """
    try:
        # Create destination directory if needed
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
        return f"Successfully moved {source} to {destination}"
    except Exception as e:
        return f"Error moving file: {str(e)}"
