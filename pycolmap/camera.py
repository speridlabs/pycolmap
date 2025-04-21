import numpy as np
from typing import Optional, Union, List
from numpy.typing import NDArray

from .types import CAMERA_MODEL_NAMES, CameraModelType 


class Camera:
    """
    Represents a camera in a COLMAP reconstruction, holding intrinsic parameters.
    NOTE: In the optimized version, instances of this class are typically created
    on-demand for accessing data and are not stored persistently in bulk.
    """

    id: int
    model: str
    width: int
    height: int
    params: NDArray[np.float32] # Shape (N,) where N is the number of parameters

    _calibration_matrix: Optional[np.ndarray] = None # Cache for K matrix

    def __init__(self, id: int, model: str, width: int, height: int, params: Union[NDArray[np.float32], List[float]]):
        """
        Initializes a Camera instance.

        Args:
            id: Unique camera identifier.
            model: Camera model name (must be a valid COLMAP model name).
            width: Image width in pixels.
            height: Image height in pixels.
            params: Numpy array or list of camera intrinsic parameters.

        Raises:
            ValueError: If the model name is unknown or the number of parameters
                        does not match the specified model.
        """
        if model not in CAMERA_MODEL_NAMES:
             raise ValueError(f"Unknown camera model name: {model}")
        if width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive integers.")

        expected_params = CAMERA_MODEL_NAMES[model].num_params

        if len(params) < expected_params:
             raise ValueError(
                 f"Camera model '{model}' expects {expected_params} parameters, "
                 f"but received array of length {len(params)}."
             )
        if len(params) > expected_params and np.any(params[expected_params:] != 0.0):
             # Only warn if non-zero extra params exist, could be padding
             print(f"Warning: Camera model '{model}' expects {expected_params} parameters, "
                  f"but {len(params)} were provided. Ignoring extra values.")


        self.id = id
        self.model = model
        self.width = width
        self.height = height
        # Ensure params is a numpy array of the correct type and store only the relevant ones
        params_array = np.array(params, dtype=np.float32)
        self.params = params_array[:expected_params]
        self._calibration_matrix = None # Invalidate cache

    def get_model_id(self) -> int:
        """Returns the numeric ID of the camera model."""
        return CAMERA_MODEL_NAMES[self.model].model_id

    def get_num_params(self) -> int:
        """Returns the number of parameters for this camera model."""
        return CAMERA_MODEL_NAMES[self.model].num_params

    def get_calibration_matrix(self) -> np.ndarray:
        """
        Calculates and returns the 3x3 camera calibration matrix (K).
        Handles various COLMAP camera models. Caches the result for efficiency.
        """
        if self._calibration_matrix is not None:
            return self._calibration_matrix.copy()

        K = np.eye(3, dtype=np.float64)
        p = self.params # This is already a numpy array

        model_type = CAMERA_MODEL_NAMES[self.model].model_id

        if model_type == CameraModelType.SIMPLE_PINHOLE.value: # f, cx, cy
            K[0, 0] = K[1, 1] = p[0]; K[0, 2] = p[1]; K[1, 2] = p[2]
        elif model_type == CameraModelType.PINHOLE.value: # fx, fy, cx, cy
            K[0, 0] = p[0]; K[1, 1] = p[1]; K[0, 2] = p[2]; K[1, 2] = p[3]
        elif model_type == CameraModelType.SIMPLE_RADIAL.value: # f, cx, cy, k1
            K[0, 0] = K[1, 1] = p[0]; K[0, 2] = p[1]; K[1, 2] = p[2]
        elif model_type == CameraModelType.RADIAL.value: # f, cx, cy, k1, k2
            K[0, 0] = K[1, 1] = p[0]; K[0, 2] = p[1]; K[1, 2] = p[2]
        elif model_type == CameraModelType.OPENCV.value: # fx, fy, cx, cy, k1, k2, p1, p2
            K[0, 0] = p[0]; K[1, 1] = p[1]; K[0, 2] = p[2]; K[1, 2] = p[3]
        elif model_type == CameraModelType.OPENCV_FISHEYE.value: # fx, fy, cx, cy, k1, k2, k3, k4
            K[0, 0] = p[0]; K[1, 1] = p[1]; K[0, 2] = p[2]; K[1, 2] = p[3]
        elif model_type == CameraModelType.FULL_OPENCV.value: # fx, fy, cx, cy, k1..k6, p1, p2
            K[0, 0] = p[0]; K[1, 1] = p[1]; K[0, 2] = p[2]; K[1, 2] = p[3]
        elif model_type == CameraModelType.FOV.value: # fx, fy, cx, cy, omega
            K[0, 0] = p[0]; K[1, 1] = p[1]; K[0, 2] = p[2]; K[1, 2] = p[3]
        elif model_type == CameraModelType.SIMPLE_RADIAL_FISHEYE.value: # f, cx, cy, k
            K[0, 0] = K[1, 1] = p[0]; K[0, 2] = p[1]; K[1, 2] = p[2]
        elif model_type == CameraModelType.RADIAL_FISHEYE.value: # f, cx, cy, k1, k2
            K[0, 0] = K[1, 1] = p[0]; K[0, 2] = p[1]; K[1, 2] = p[2]
        elif model_type == CameraModelType.THIN_PRISM_FISHEYE.value: # fx, fy, cx, cy, k1..k4, p1, p2, sx1, sy1
            K[0, 0] = p[0]; K[1, 1] = p[1]; K[0, 2] = p[2]; K[1, 2] = p[3]
        else:
             # Should not happen due to init check
             raise NotImplementedError(f"Calibration matrix calculation for model '{self.model}' not implemented.")

        self._calibration_matrix = K
        return K.copy() # Return copy

    def get_distortion_params(self) -> Optional[np.ndarray]:
        """
        Extracts and returns the distortion parameters as a NumPy array.
        Returns None if the model has no distortion parameters.
        """
        p = self.params
        model_type = CAMERA_MODEL_NAMES[self.model].model_id

        if model_type in (CameraModelType.SIMPLE_PINHOLE.value, CameraModelType.PINHOLE.value):
            return None
        elif model_type in (CameraModelType.SIMPLE_RADIAL.value, CameraModelType.SIMPLE_RADIAL_FISHEYE.value):
            return p[3:] # k1 or k
        elif model_type in (CameraModelType.RADIAL.value, CameraModelType.RADIAL_FISHEYE.value):
            return p[3:] # k1, k2
        elif model_type == CameraModelType.OPENCV.value:
            return p[4:] # k1, k2, p1, p2
        elif model_type == CameraModelType.OPENCV_FISHEYE.value:
            return p[4:] # k1, k2, k3, k4
        elif model_type == CameraModelType.FULL_OPENCV.value:
            return p[4:] # k1..k6, p1, p2
        elif model_type == CameraModelType.FOV.value:
            return p[4:] # omega
        elif model_type == CameraModelType.THIN_PRISM_FISHEYE.value:
            return p[4:] # k1..k4, p1, p2, sx1, sy1
        else:
            # Should not happen
             raise NotImplementedError(f"Distortion parameter extraction for model '{self.model}' not implemented.")

    def has_distortion(self) -> bool:
        """Checks if the camera model includes distortion parameters."""
        model_type = CAMERA_MODEL_NAMES[self.model].model_id
        return model_type not in (CameraModelType.SIMPLE_PINHOLE.value, CameraModelType.PINHOLE.value)

    def __repr__(self) -> str:
        params_str = np.array2string(self.params, precision=3, separator=', ', suppress_small=True)
        return (f"Camera(id={self.id}, model='{self.model}', "
                f"width={self.width}, height={self.height}, "
                f"params={params_str})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Camera):
            return NotImplemented
        # Check relevant parameters only using get_num_params
        num_params_self = self.get_num_params()
        num_params_other = other.get_num_params()
        return self.id == other.id and \
               self.model == other.model and \
               self.width == other.width and \
               self.height == other.height and \
               num_params_self == num_params_other and \
               np.allclose(self.params[:num_params_self], other.params[:num_params_other])

    def __hash__(self) -> int:
        # Hash based on immutable or effectively immutable properties
        return hash((self.id, self.model, self.width, self.height, self.params.tobytes()))
