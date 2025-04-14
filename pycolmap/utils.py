import os
import numpy as np
from typing import Tuple, Optional

from .image import Image
from .camera import Camera


def detect_model_format(path: str) -> str:
    """Detect COLMAP model format in a directory.
    
    Args:
        path: Directory containing the model files
        
    Returns:
        File extension ('.bin' or '.txt') if detected, empty string otherwise
    """
    if os.path.isfile(os.path.join(path, "cameras.bin")) and \
       os.path.isfile(os.path.join(path, "images.bin")) and \
       os.path.isfile(os.path.join(path, "points3D.bin")):
        return ".bin"
    
    if os.path.isfile(os.path.join(path, "cameras.txt")) and \
       os.path.isfile(os.path.join(path, "images.txt")) and \
       os.path.isfile(os.path.join(path, "points3D.txt")):
        return ".txt"
    
    return ""


def find_model_path(base_path: str) -> Optional[str]:
    """Find a COLMAP model in common directories.
    
    Args:
        base_path: Base directory to search in
        
    Returns:
        Path to the directory containing the model files, or None if not found
    """
    # Common model locations
    candidates = [
        os.path.join(base_path, "sparse", "0"),
        os.path.join(base_path, "sparse"),
        base_path
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate) and detect_model_format(candidate):
            return candidate
    
    return None


def qvec2rotmat(qvec: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        qvec: Quaternion as (w, x, y, z)
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = qvec
    
    R = np.zeros((3, 3))
    
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


def rotmat2qvec(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as (w, x, y, z)
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return (w, x, y, z)


def get_projection_matrix(camera: Camera, image: Image) -> np.ndarray:
    """Get the projection matrix for a camera-image pair.
    
    Args:
        camera: Camera object
        image: Image object
        
    Returns:
        3x4 projection matrix
    """
    K = camera.get_calibration_matrix()
    R = image.get_rotation_matrix()
    t = np.array(image.tvec).reshape(3, 1)
    
    # Projection matrix: P = K[R|t]
    Rt = np.hstack((R, t))
    P = K @ Rt
    
    return P


def triangulate_point(P1: np.ndarray, P2: np.ndarray, 
                      point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> Tuple[float, float, float]:
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
    
    # Convert from homogeneous to Euclidean coordinates
    X = X / X[3]
    
    return (X[0], X[1], X[2])


def angle_between_rays(camera1_center: Tuple[float, float, float],
                       camera2_center: Tuple[float, float, float],
                       point3D: Tuple[float, float, float]) -> float:
    """Calculate the angle between two camera rays to a 3D point.
    
    Args:
        camera1_center: First camera center (x, y, z)
        camera2_center: Second camera center (x, y, z)
        point3D: 3D point (x, y, z)
        
    Returns:
        Angle in degrees
    """
    # Convert to numpy arrays
    np_camera1_center = np.array(camera1_center)
    np_camera2_center = np.array(camera2_center)
    np_point3D = np.array(point3D)
    
    # Calculate rays
    ray1 = np_point3D - np_camera1_center
    ray2 = np_point3D - np_camera2_center
    
    # Normalize
    ray1 = ray1 / np.linalg.norm(ray1)
    ray2 = ray2 / np.linalg.norm(ray2)
    
    # Calculate angle
    cos_angle = np.clip(np.dot(ray1, ray2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg
