from enum import Enum

# A special value representing an invalid point3D ID
INVALID_POINT3D_ID = - 1

class CameraModelType(Enum):
    """Enumeration of camera model types supported by COLMAP."""
    SIMPLE_PINHOLE = 0
    PINHOLE = 1
    SIMPLE_RADIAL = 2
    RADIAL = 3
    OPENCV = 4
    OPENCV_FISHEYE = 5
    FULL_OPENCV = 6
    FOV = 7
    SIMPLE_RADIAL_FISHEYE = 8
    RADIAL_FISHEYE = 9
    THIN_PRISM_FISHEYE = 10

class CameraModel:
    """Camera model information."""
    
    model_id: int
    model_name: str
    num_params: int
    
    def __init__(self, model_id: int, model_name: str, num_params: int):
        """Initialize a camera model.
        
        Args:
            model_id: Numeric ID of the camera model
            model_name: String name of the camera model
            num_params: Number of parameters for this model
        """
        self.model_id = model_id
        self.model_name = model_name
        self.num_params = num_params

CAMERA_MODELS = [
    CameraModel(CameraModelType.SIMPLE_PINHOLE.value, "SIMPLE_PINHOLE", 3),
    CameraModel(CameraModelType.PINHOLE.value, "PINHOLE", 4),
    CameraModel(CameraModelType.SIMPLE_RADIAL.value, "SIMPLE_RADIAL", 4),
    CameraModel(CameraModelType.RADIAL.value, "RADIAL", 5),
    CameraModel(CameraModelType.OPENCV.value, "OPENCV", 8),
    CameraModel(CameraModelType.OPENCV_FISHEYE.value, "OPENCV_FISHEYE", 8),
    CameraModel(CameraModelType.FULL_OPENCV.value, "FULL_OPENCV", 12),
    CameraModel(CameraModelType.FOV.value, "FOV", 5),
    CameraModel(CameraModelType.SIMPLE_RADIAL_FISHEYE.value, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(CameraModelType.RADIAL_FISHEYE.value, "RADIAL_FISHEYE", 5),
    CameraModel(CameraModelType.THIN_PRISM_FISHEYE.value, "THIN_PRISM_FISHEYE", 12)
]

MAX_CAMERA_PARAMS = max(model.num_params for model in CAMERA_MODELS)
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}
