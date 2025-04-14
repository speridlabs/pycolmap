import os
from typing import Dict, Tuple

from ..image import Image
from ..camera import Camera
from ..point3d import Point3D

from .text import read_text_model, write_text_model
from .binary import read_binary_model, write_binary_model


def read_model(path: str, ext: str = "") -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    """Read a COLMAP reconstruction from a directory.
    
    Args:
        path: Directory containing the reconstruction
        ext: Optional file extension ('.bin' or '.txt')
        
    Returns:
        Tuple of (cameras, images, points3D) dictionaries
    """
    if ext == "":
        # Try to detect the format automatically
        if all(p.endswith(".bin") for p in [
            f"{path}/cameras.bin", 
            f"{path}/images.bin", 
            f"{path}/points3D.bin"
        ] if os.path.exists(p)):
            ext = ".bin"
        elif all(p.endswith(".txt") for p in [
            f"{path}/cameras.txt", 
            f"{path}/images.txt", 
            f"{path}/points3D.txt"
        ] if os.path.exists(p)):
            ext = ".txt"
        else:
            raise ValueError("Could not detect model format. Provide '.bin' or '.txt'")
    
    if ext == ".bin":
        return read_binary_model(path)
    elif ext == ".txt":
        return read_text_model(path)
    else:
        raise ValueError(f"Unknown model format: {ext}")


def write_model(cameras: Dict[int, Camera], 
                images: Dict[int, Image], 
                points3D: Dict[int, Point3D], 
                path: str, 
                binary: bool = True) -> None:
    """Write a COLMAP reconstruction to a directory.
    
    Args:
        cameras: Dictionary of Camera objects
        images: Dictionary of Image objects
        points3D: Dictionary of Point3D objects
        path: Output directory
        binary: Whether to use binary format (True) or text format (False)
    """
    import os
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if binary:
        write_binary_model(cameras, images, points3D, path)
    else:
        write_text_model(cameras, images, points3D, path)
