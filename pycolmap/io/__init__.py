import os
from typing import Tuple, Optional, Union, TYPE_CHECKING

from ..utils import detect_model_format
from .data import CameraData, ImageData, Point3DData

from .text import read_text_model, write_text_model
from .binary import read_binary_model, write_binary_model

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from ..reconstruction import ColmapReconstruction

# Define a type alias for the internal data tuple
InternalReconstructionData = Tuple[CameraData, ImageData, Point3DData]

# Expose key functions directly under io
# Note: The exposed read/write functions might need adjustment
# depending on whether they should return objects or internal structures.
# Keeping the `read_model` and `write_model` as the primary interface is best.
__all__ = [
    "read_model",
    "write_model",
]


def read_model(path: str, only_3d_features: bool, file_format: Optional[str] = None) -> InternalReconstructionData:
    """
    Reads a COLMAP reconstruction model from a specified directory
    into internal NumPy-based data structures.

    Automatically detects the format (binary '.bin' or text '.txt') if not
    explicitly provided.

    Args:
        path: Directory containing model files (cameras, images, points3D).
        only_3d_features: If True, discards 2D features in images that don't
                          correspond to a 3D point (point3D_id == -1).
        file_format: Optional explicit format ('.bin' or '.txt').

    Returns:
        A tuple containing internal data structures:
        (CameraData, ImageData, Point3DData)

    Raises:
        FileNotFoundError: If path invalid or essential files missing.
        ValueError: If format unknown or invalid.
        EOFError: If binary files are truncated.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Input path is not a valid directory: {path}")

    resolved_format = file_format
    if resolved_format is None:
        resolved_format = detect_model_format(path)
        if not resolved_format:
            raise ValueError(f"Could not auto-detect COLMAP model format in '{path}'.")

    required_files_map = {
        ".bin": ["cameras.bin", "images.bin", "points3D.bin"],
        ".txt": ["cameras.txt", "images.txt", "points3D.txt"],
    }

    if resolved_format not in required_files_map:
        raise ValueError(f"Unsupported format '{resolved_format}'. Use '.bin' or '.txt'.")

    # Check for required files
    required_files = required_files_map[resolved_format]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing required {resolved_format} model file(s) in '{path}': {', '.join(missing_files)}")

    if resolved_format == ".bin":
        return read_binary_model(path, only_3d_features=only_3d_features)
    else: # resolved_format == ".txt":
        return read_text_model(path, only_3d_features=only_3d_features)


def write_model(
    data: Union[InternalReconstructionData, 'ColmapReconstruction'], # Accept internal data or a Reconstruction object
    output_path: str,
    binary: bool = True
) -> None:
    """
    Writes a COLMAP reconstruction model to a specified directory.

    Args:
        data: Either a tuple of internal data structures (CameraData, ImageData, Point3DData)
              or a ColmapReconstruction object.
        output_path: The directory where the model files will be saved.
        binary: If True, save in binary ('.bin') format; otherwise, text ('.txt').
    """
    os.makedirs(output_path, exist_ok=True)

    images_data: ImageData
    cameras_data: CameraData
    points3D_data: Point3DData

    # Need to import ColmapReconstruction locally to avoid circular dependency at module level
    from ..reconstruction import ColmapReconstruction
    if isinstance(data, ColmapReconstruction):
        cameras_data, images_data, points3D_data = data.get_internal_data()
    elif isinstance(data, tuple) and len(data) == 3 and \
         isinstance(data[0], CameraData) and \
         isinstance(data[1], ImageData) and \
         isinstance(data[2], Point3DData):
        cameras_data, images_data, points3D_data = data
    else:
        raise TypeError("Input 'data' must be a ColmapReconstruction object or a tuple (CameraData, ImageData, Point3DData)")

    if binary: return write_binary_model(cameras_data, images_data, points3D_data, output_path)
    return write_text_model(cameras_data, images_data, points3D_data, output_path)

