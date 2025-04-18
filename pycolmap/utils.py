import os
import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

# Avoid circular import for type hinting
if TYPE_CHECKING:
    from .image import Image
    from .camera import Camera


def detect_model_format(path: str) -> str:
    """Detect COLMAP model format in a directory."""
    if not os.path.isdir(path):
        return ""

    has_bin = (os.path.isfile(os.path.join(path, "cameras.bin")) and
               os.path.isfile(os.path.join(path, "images.bin")) and
               os.path.isfile(os.path.join(path, "points3D.bin")))
    if has_bin:
        return ".bin"

    has_txt = (os.path.isfile(os.path.join(path, "cameras.txt")) and
               os.path.isfile(os.path.join(path, "images.txt")) and
               os.path.isfile(os.path.join(path, "points3D.txt")))
    if has_txt:
        return ".txt"

    return ""


def find_model_path(base_path: str) -> Optional[str]:
    """Find a COLMAP model in common directories.
    
    Args:
        base_path: Base directory to search in
        
    Returns:
        Path to the directory containing the model files, or None if not found
    """
    # common model locations
    candidates = [
        os.path.join(base_path, "sparse", "0"),
        os.path.join(base_path, "sparse"),
        base_path
    ]

    for candidate in candidates:
        if os.path.exists(candidate) and detect_model_format(candidate):
            return candidate

    return None


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        qvec: Quaternion as (w, x, y, z)
        
    Returns:
        3x3 rotation matrix
    """
    if qvec.shape != (4,):
        raise ValueError("qvec must have shape (4,)")

    w, x, y, z = qvec
    R = np.zeros((3, 3), dtype=np.float64)

    R[0, 0] = 1 - 2 * y**2 - 2 * z**2
    R[0, 1] = 2 * x * y - 2 * w * z
    R[0, 2] = 2 * x * z + 2 * w * y

    R[1, 0] = 2 * x * y + 2 * w * z
    R[1, 1] = 1 - 2 * x**2 - 2 * z**2
    R[1, 2] = 2 * y * z - 2 * w * x

    R[2, 0] = 2 * x * z - 2 * w * y
    R[2, 1] = 2 * y * z + 2 * w * x
    R[2, 2] = 1 - 2 * x**2 - 2 * y**2

    return R


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as (w, x, y, z)
    """

    if R.shape != (3, 3):
        raise ValueError("R must have shape (3, 3)")

    trace = np.trace(R)
    q = np.zeros(4, dtype=np.float64)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s # w
        q[1] = (R[2, 1] - R[1, 2]) * s # x
        q[2] = (R[0, 2] - R[2, 0]) * s # y
        q[3] = (R[1, 0] - R[0, 1]) * s # z
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s

    return q


def get_projection_matrix(camera: 'Camera', image: 'Image') -> np.ndarray:
    """Get the projection matrix for a camera-image pair.
    
    Args:
        camera: Camera object
        image: Image object
        
    Returns:
        3x4 projection matrix
    """

    K = camera.get_calibration_matrix()
    R = image.get_rotation_matrix() # Uses image's method
    t = image.tvec.reshape(3, 1)    # Use image's tvec directly

    # Projection matrix: P = K[R|t]
    Rt = np.hstack((R, t))
    P = K @ Rt
    return P


def triangulate_point(P1: np.ndarray, P2: np.ndarray,
                      point1: Tuple[float, float],
                      point2: Tuple[float, float]) -> np.ndarray:
    """Triangulate a 3D point from two corresponding 2D points.
    
    Args:
        P1: 3x4 projection matrix for first camera
        P2: 3x4 projection matrix for second camera
        point1: 2D point in first image (x, y)
        point2: 2D point in second image (x, y)
        
    Returns:
        Triangulated 3D point (x, y, z)
    """

    # Build system of equations
    A = np.zeros((4, 4))

    x1, y1 = point1
    x2, y2 = point2

    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]

    # Solve using SVD
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]

    X = X / X[3] # Homogeneous to Euclidean
    return X[:3] # Return as (3,) numpy array


def angle_between_rays(camera1_center: np.ndarray,
                       camera2_center: np.ndarray,
                       point3D: np.ndarray) -> float:
    """Calculate the angle between two camera rays to a 3D point.
    
    Args:
        camera1_center: First camera center (x, y, z)
        camera2_center: Second camera center (x, y, z)
        point3D: 3D point (x, y, z)
        
    Returns:
        Angle in degrees
    """
    if camera1_center.shape != (3,) or camera2_center.shape != (3,) or point3D.shape != (3,):
        raise ValueError("All inputs must be numpy arrays of shape (3,)")

    # Calculate rays
    ray1 = point3D - camera1_center
    ray2 = point3D - camera2_center

    # Normalize
    norm1 = np.linalg.norm(ray1)
    norm2 = np.linalg.norm(ray2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Camera centers and point3D must not be the same")

    # Normalize
    ray1 /= norm1
    ray2 /= norm2

    # Calculate angle
    cos_angle = np.clip(np.dot(ray1, ray2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    # Convert to degrees
    return np.degrees(angle_rad)
