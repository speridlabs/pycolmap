__version__ = "0.1.1" # Increment version

__all__ = [
    # Core classes
    "Camera",
    "Image",
    "Point3D",
    "ColmapReconstruction",
    # Types & Constants
    "CameraModel",
    "CameraModelType",
    "CAMERA_MODELS",
    "CAMERA_MODEL_IDS",
    "CAMERA_MODEL_NAMES",
    "INVALID_POINT3D_ID",
    # IO Functions
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
    # Utility functions
    "qvec2rotmat",
    "rotmat2qvec",
    "angle_between_rays",
    "find_model_path",
    "detect_model_format",
]

from .camera import Camera
from .image import Image
from .point3d import Point3D
from .reconstruction import ColmapReconstruction
from .types import (
    CameraModel,
    CameraModelType,
    CAMERA_MODELS,
    CAMERA_MODEL_IDS,
    CAMERA_MODEL_NAMES,
    INVALID_POINT3D_ID,
)
from .io import (
    read_model,
    write_model,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
    read_cameras_text,
    read_images_text,
    read_points3D_text,
    write_cameras_text,
    write_images_text,
    write_points3D_text,
)
from .utils import (
    qvec2rotmat,
    rotmat2qvec,
    angle_between_rays,
    find_model_path,
    detect_model_format,
)

# Suppress numpy warnings (optional, use with caution)
# import warnings
# import numpy as np
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
