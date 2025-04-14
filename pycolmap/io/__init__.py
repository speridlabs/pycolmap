import os
from typing import Dict, Tuple, Optional

from ..image import Image
from ..camera import Camera
from ..point3d import Point3D
from ..utils import detect_model_format # Use utils for detection

# Import specific functions for clarity and potential selective use
from .text import (
    read_cameras_text, read_images_text, read_points3D_text,
    write_cameras_text, write_images_text, write_points3D_text,
    read_text_model, write_text_model
)
from .binary import (
    read_cameras_binary, read_images_binary, read_points3D_binary,
    write_cameras_binary, write_images_binary, write_points3D_binary,
    read_binary_model, write_binary_model
)

# Expose key functions directly under io
__all__ = [
    "read_model",
    "write_model",
    "read_cameras_binary",
    "read_images_binary",
    "read_points3D_binary",
    "write_cameras_binary",
    "write_images_binary",
    "write_points3D_binary",
    "read_cameras_text",
    "read_images_text",
    "read_points3D_text",
    "write_cameras_text",
    "write_images_text",
    "write_points3D_text",
]


def read_model(path: str, file_format: Optional[str] = None) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    """
    Reads a COLMAP reconstruction model from a specified directory.

    Automatically detects the format (binary '.bin' or text '.txt') if not
    explicitly provided.

    Args:
        path: The directory containing the COLMAP model files
              (cameras, images, points3D).
        file_format: Optional explicit format specifier ('.bin' or '.txt').
                     If None, the format is auto-detected. Defaults to None.

    Returns:
        A tuple containing three dictionaries:
        - cameras: Mapping camera_id (int) to Camera objects.
        - images: Mapping image_id (int) to Image objects.
        - points3D: Mapping point3D_id (int) to Point3D objects.

    Raises:
        FileNotFoundError: If the specified path does not exist or if essential
                         model files are missing in the detected/specified format.
        ValueError: If the format cannot be detected and is not provided, or if
                    an invalid format is specified.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input path is not a valid directory: {path}")

    resolved_format = file_format
    if resolved_format is None:
        resolved_format = detect_model_format(path)
        if resolved_format is None:
            raise ValueError(
                f"Could not automatically detect COLMAP model format in '{path}'. "
                "Please ensure cameras, images, and points3D files exist "
                "with either '.bin' or '.txt' extensions, or specify the "
                "format explicitly using the 'file_format' argument."
            )

    if resolved_format == ".bin":
        # Check for required binary files before attempting read
        required_files = ["cameras.bin", "images.bin", "points3D.bin"]
        missing_files = [f for f in required_files if not os.path.isfile(os.path.join(path, f))]
        if missing_files:
            raise FileNotFoundError(f"Missing required binary model file(s) in '{path}': {', '.join(missing_files)}")
        return read_binary_model(path)

    elif resolved_format == ".txt":
         # Check for required text files before attempting read
        required_files = ["cameras.txt", "images.txt", "points3D.txt"]
        missing_files = [f for f in required_files if not os.path.isfile(os.path.join(path, f))]
        if missing_files:
            raise FileNotFoundError(f"Missing required text model file(s) in '{path}': {', '.join(missing_files)}")
        return read_text_model(path)

    else:
        raise ValueError(f"Unsupported model format specified: '{resolved_format}'. Use '.bin' or '.txt'.")


def write_model(cameras: Dict[int, Camera],
                images: Dict[int, Image],
                points3D: Dict[int, Point3D],
                output_path: str,
                binary: bool = True) -> None:
    """
    Writes a COLMAP reconstruction model to a specified directory.

    Args:
        cameras: A dictionary mapping camera_id to Camera objects.
        images: A dictionary mapping image_id to Image objects.
        points3D: A dictionary mapping point3D_id to Point3D objects.
        output_path: The directory where the model files will be saved.
                     The directory will be created if it does not exist.
        binary: If True, saves the model in binary ('.bin') format.
                If False, saves in text ('.txt') format. Defaults to True.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if binary:
        write_binary_model(cameras, images, points3D, output_path)
    else:
        write_text_model(cameras, images, points3D, output_path)
