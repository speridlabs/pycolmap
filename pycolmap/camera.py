import numpy as np
from typing import List, Tuple 

from .types import CAMERA_MODEL_NAMES


class Camera:
    """Camera intrinsic parameters."""
    
    id: int
    model: str
    width: int
    height: int
    params: List[float]
    
    def __init__(self, id: int, model: str, width: int, height: int, params: List[float]):
        """Initialize a camera.
        
        Args:
            id: Unique camera identifier
            model: Camera model name (e.g., 'PINHOLE', 'OPENCV')
            width: Image width in pixels
            height: Image height in pixels
            params: Camera parameters (varies by model)
        """
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

        # Validate parameters
        if model in CAMERA_MODEL_NAMES:
            expected_params = CAMERA_MODEL_NAMES[model].num_params
            if len(params) != expected_params:
                raise ValueError(f"Camera model {model} expects {expected_params} parameters, got {len(params)}")
    
    def get_calibration_matrix(self) -> np.ndarray:
        """Get the camera calibration matrix (K).
        
        Returns:
            3x3 calibration matrix
        """
        K = np.eye(3)
        
        if self.model == 'SIMPLE_PINHOLE':
            # f, cx, cy
            K[0, 0] = self.params[0]  # f
            K[1, 1] = self.params[0]  # f
            K[0, 2] = self.params[1]  # cx
            K[1, 2] = self.params[2]  # cy
            
        elif self.model == 'PINHOLE':
            # fx, fy, cx, cy
            K[0, 0] = self.params[0]  # fx
            K[1, 1] = self.params[1]  # fy
            K[0, 2] = self.params[2]  # cx
            K[1, 2] = self.params[3]  # cy
            
        elif self.model == 'OPENCV' or self.model == 'OPENCV_FISHEYE':
            # fx, fy, cx, cy, ...
            K[0, 0] = self.params[0]  # fx
            K[1, 1] = self.params[1]  # fy
            K[0, 2] = self.params[2]  # cx
            K[1, 2] = self.params[3]  # cy
            
        elif self.model == 'SIMPLE_RADIAL' or self.model == 'RADIAL' or self.model == 'SIMPLE_RADIAL_FISHEYE' or self.model == 'RADIAL_FISHEYE':
            # f, cx, cy, k1, ...
            K[0, 0] = self.params[0]  # f
            K[1, 1] = self.params[0]  # f
            K[0, 2] = self.params[1]  # cx
            K[1, 2] = self.params[2]  # cy
            
        else:
            raise NotImplementedError(f"Calibration matrix for camera model {self.model} not implemented")
            
        return K
    
    def get_distortion_params(self) -> List[float]:
        """Get distortion parameters.
        
        Returns:
            List of distortion parameters (model-dependent)
        """
        if self.model == 'SIMPLE_PINHOLE' or self.model == 'PINHOLE':
            return []
        elif self.model == 'SIMPLE_RADIAL':
            return [self.params[3]]  # k1
        elif self.model == 'RADIAL':
            return [self.params[3], self.params[4]]  # k1, k2
        elif self.model == 'OPENCV':
            return self.params[4:8]  # k1, k2, p1, p2
        elif self.model == 'OPENCV_FISHEYE':
            return self.params[4:8]  # k1, k2, k3, k4
        else:
            raise NotImplementedError(f"Distortion parameters for camera model {self.model} not implemented")

    def has_distortion(self) -> bool:
        """Check if camera has distortion parameters.
        
        Returns:
            True if camera model includes distortion, False otherwise
        """
        return len(self.get_distortion_params()) > 0
        
    def __repr__(self) -> str:
        return f"Camera(id={self.id}, model={self.model}, width={self.width}, height={self.height})"
