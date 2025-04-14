import numpy as np
from typing import List, Tuple, Optional

from .types import CAMERA_MODEL_NAMES, CameraModelType


class Camera:
    """
    Represents a camera in a COLMAP reconstruction, holding intrinsic parameters.

    Attributes:
        id: Unique camera identifier.
        model: Camera model name (must be a valid COLMAP model name).
        width: Image width in pixels.
        height: Image height in pixels.
        params: List of camera intrinsic parameters.
    """

    id: int
    model: str
    width: int
    height: int
    params: List[float]

    _calibration_matrix: Optional[np.ndarray] = None # Cache for K matrix

    def __init__(self, id: int, model: str, width: int, height: int, params: List[float]):
        """
        Initializes a Camera instance.

        Args:
            id: Unique camera identifier.
            model: Camera model name (must be a valid COLMAP model name).
            width: Image width in pixels.
            height: Image height in pixels.
            params: List of camera intrinsic parameters.

        Raises:
            ValueError: If the model name is unknown or the number of parameters
                        does not match the specified model.
        """
        if model not in CAMERA_MODEL_NAMES:
             raise ValueError(f"Unknown camera model name: {model}")
        if width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive integers.")

        expected_params = CAMERA_MODEL_NAMES[model].num_params
        if len(params) != expected_params:
            raise ValueError(
                f"Camera model '{model}' expects {expected_params} parameters, "
                f"but {len(params)} were provided."
            )

        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params
        self._calibration_matrix = None # Invalidate cache

    def get_calibration_matrix(self) -> np.ndarray:
        """
        Calculates and returns the 3x3 camera calibration matrix (K).

        Handles various COLMAP camera models. Caches the result for efficiency.

        Returns:
            A 3x3 numpy array representing the calibration matrix.

        Raises:
            NotImplementedError: If the calibration matrix calculation for the
                                 camera's model is not supported.
        """
        if self._calibration_matrix is not None:
            return self._calibration_matrix.copy() # Return copy to prevent external modification

        K = np.eye(3, dtype=np.float64)
        p = self.params

        if self.model == CameraModelType.SIMPLE_PINHOLE.name: # f, cx, cy
            K[0, 0] = K[1, 1] = p[0] # fx = fy = f
            K[0, 2] = p[1]          # cx
            K[1, 2] = p[2]          # cy
        elif self.model == CameraModelType.PINHOLE.name: # fx, fy, cx, cy
            K[0, 0] = p[0]          # fx
            K[1, 1] = p[1]          # fy
            K[0, 2] = p[2]          # cx
            K[1, 2] = p[3]          # cy
        elif self.model == CameraModelType.SIMPLE_RADIAL.name: # f, cx, cy, k1
            K[0, 0] = K[1, 1] = p[0] # fx = fy = f
            K[0, 2] = p[1]          # cx
            K[1, 2] = p[2]          # cy
        elif self.model == CameraModelType.RADIAL.name: # f, cx, cy, k1, k2
            K[0, 0] = K[1, 1] = p[0] # fx = fy = f
            K[0, 2] = p[1]          # cx
            K[1, 2] = p[2]          # cy
        elif self.model == CameraModelType.OPENCV.name: # fx, fy, cx, cy, k1, k2, p1, p2
            K[0, 0] = p[0]          # fx
            K[1, 1] = p[1]          # fy
            K[0, 2] = p[2]          # cx
            K[1, 2] = p[3]          # cy
        elif self.model == CameraModelType.OPENCV_FISHEYE.name: # fx, fy, cx, cy, k1, k2, k3, k4
             K[0, 0] = p[0]          # fx
             K[1, 1] = p[1]          # fy
             K[0, 2] = p[2]          # cx
             K[1, 2] = p[3]          # cy
        elif self.model == CameraModelType.FULL_OPENCV.name: # fx, fy, cx, cy, k1..k6, p1, p2
             K[0, 0] = p[0]          # fx
             K[1, 1] = p[1]          # fy
             K[0, 2] = p[2]          # cx
             K[1, 2] = p[3]          # cy
        elif self.model == CameraModelType.FOV.name: # fx, fy, cx, cy, omega
             K[0, 0] = p[0]          # fx
             K[1, 1] = p[1]          # fy
             K[0, 2] = p[2]          # cx
             K[1, 2] = p[3]          # cy
        elif self.model == CameraModelType.SIMPLE_RADIAL_FISHEYE.name: # f, cx, cy, k
             K[0, 0] = K[1, 1] = p[0] # fx = fy = f
             K[0, 2] = p[1]          # cx
             K[1, 2] = p[2]          # cy
        elif self.model == CameraModelType.RADIAL_FISHEYE.name: # f, cx, cy, k1, k2
             K[0, 0] = K[1, 1] = p[0] # fx = fy = f
             K[0, 2] = p[1]          # cx
             K[1, 2] = p[2]          # cy
        elif self.model == CameraModelType.THIN_PRISM_FISHEYE.name: # fx, fy, cx, cy, k1..k4, p1, p2, sx1, sy1
             K[0, 0] = p[0]          # fx
             K[1, 1] = p[1]          # fy
             K[0, 2] = p[2]          # cx
             K[1, 2] = p[3]          # cy
        else:
             raise NotImplementedError(
                 f"Calibration matrix calculation for camera model '{self.model}' is not implemented."
             )

        self._calibration_matrix = K
        return K.copy() # Return copy

    def get_distortion_params(self) -> List[float]:
        """
        Extracts and returns the distortion parameters from the camera's parameter list.

        The number and meaning of distortion parameters depend on the camera model.

        Returns:
            A list containing the distortion parameters, or an empty list if the
            model has no distortion (e.g., PINHOLE).

        Raises:
            NotImplementedError: If distortion parameter extraction for the
                                 camera's model is not supported.
        """
        p = self.params
        if self.model == CameraModelType.SIMPLE_PINHOLE.name or \
           self.model == CameraModelType.PINHOLE.name:
            return []
        elif self.model == CameraModelType.SIMPLE_RADIAL.name: # f, cx, cy, k1
            return p[3:] # k1
        elif self.model == CameraModelType.RADIAL.name: # f, cx, cy, k1, k2
            return p[3:] # k1, k2
        elif self.model == CameraModelType.OPENCV.name: # fx, fy, cx, cy, k1, k2, p1, p2
            return p[4:] # k1, k2, p1, p2
        elif self.model == CameraModelType.OPENCV_FISHEYE.name: # fx, fy, cx, cy, k1, k2, k3, k4
            return p[4:] # k1, k2, k3, k4
        elif self.model == CameraModelType.FULL_OPENCV.name: # fx, fy, cx, cy, k1..k6, p1, p2
            return p[4:] # k1, k2, k3, k4, k5, k6, p1, p2
        elif self.model == CameraModelType.FOV.name: # fx, fy, cx, cy, omega
            return p[4:] # omega
        elif self.model == CameraModelType.SIMPLE_RADIAL_FISHEYE.name: # f, cx, cy, k
            return p[3:] # k1
        elif self.model == CameraModelType.RADIAL_FISHEYE.name: # f, cx, cy, k1, k2
            return p[3:] # k1, k2
        elif self.model == CameraModelType.THIN_PRISM_FISHEYE.name: # fx, fy, cx, cy, k1..k4, p1, p2, sx1, sy1
            return p[4:] # k1, k2, k3, k4, p1, p2, sx1, sy1
        else:
            # Should not happen due to init check, but defensive programming
            raise NotImplementedError(
                f"Distortion parameter extraction for camera model '{self.model}' is not implemented."
            )

    def has_distortion(self) -> bool:
        """
        Checks if the camera model includes distortion parameters.

        Returns:
            True if the model has distortion parameters, False otherwise.
        """
        # Models without distortion params explicitly listed
        no_distortion_models = {
            CameraModelType.SIMPLE_PINHOLE.name,
            CameraModelType.PINHOLE.name
        }
        return self.model not in no_distortion_models

    def __repr__(self) -> str:
        return (f"Camera(id={self.id}, model='{self.model}', "
                f"width={self.width}, height={self.height}, "
                f"params=[{', '.join(f'{p:.3f}' for p in self.params)}])")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Camera):
            return NotImplemented
        return self.id == other.id and \
               self.model == other.model and \
               self.width == other.width and \
               self.height == other.height and \
               np.allclose(self.params, other.params) # Use numpy for float comparison

    def __hash__(self) -> int:
         # Hash based on immutable or effectively immutable properties
         # Convert params list to tuple for hashing
        return hash((self.id, self.model, self.width, self.height, tuple(self.params)))
