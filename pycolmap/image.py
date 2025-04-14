"""
Image classes and utilities for COLMAP reconstructions.
"""

from typing import List, Tuple, Optional
import numpy as np


class Image:
    """Image extrinsic parameters and features."""
    
    id: int
    name: str
    camera_id: int

    xys: List[Tuple[float, float]]  # List of 2D points
    point3D_ids: List[int]  # List of corresponding 3D point IDs

    tvec: Tuple[float, float, float]  # [x, y, z]
    qvec: Tuple[float, float, float, float]  # [w, x, y, z]

    
    def __init__(self, id: int, name: str, camera_id: int, 
                 qvec: Tuple[float, float, float, float], 
                 tvec: Tuple[float, float, float],
                 xys: Optional[List[Tuple[float, float]]] = None,
                 point3D_ids: Optional[List[int]] = None):
        """Initialize an image.
        
        Args:
            id: Unique image identifier
            name: Image file name
            camera_id: ID of the camera used for this image
            qvec: Quaternion rotation in [w, x, y, z] format
            tvec: Translation vector [x, y, z]
            xys: List of 2D feature points
            point3D_ids: List of corresponding 3D point IDs
        """
        self.id = id
        self.name = name
        self.camera_id = camera_id
        self.qvec = qvec
        self.tvec = tvec
        self.xys = xys if xys is not None else []
        self.point3D_ids = point3D_ids if point3D_ids is not None else []
        
        if len(self.xys) != len(self.point3D_ids):
            raise ValueError(f"Number of 2D points ({len(self.xys)}) does not match number of 3D point IDs ({len(self.point3D_ids)})")

    def qvec2rotmat(self) -> np.ndarray:
        """Convert quaternion vector to rotation matrix.
        
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = self.qvec
        
        # Convert quaternion to rotation matrix
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
        
    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from quaternion.
        
        Returns:
            3x3 rotation matrix
        """
        return self.qvec2rotmat()
        
    def get_world_to_camera_matrix(self) -> np.ndarray:
        """Get world-to-camera transformation matrix.
        
        Returns:
            4x4 transformation matrix
        """
        R = self.qvec2rotmat()
        t = np.array(self.tvec).reshape(3, 1)
        
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t.ravel()
        
        return transform
        
    def get_camera_to_world_matrix(self) -> np.ndarray:
        """Get camera-to-world transformation matrix.
        
        Returns:
            4x4 transformation matrix
        """
        world_to_camera = self.get_world_to_camera_matrix()
        return np.linalg.inv(world_to_camera)
        
    def get_camera_center(self) -> Tuple[float, float, float]:
        """Get camera center in world coordinates.
        
        Returns:
            Camera center as (x, y, z)
        """
        R = self.qvec2rotmat()
        t = np.array(self.tvec)
        center = -R.T @ t
        return (center[0], center[1], center[2])
        
    def __repr__(self) -> str:
        return f"Image(id={self.id}, name={self.name}, camera_id={self.camera_id}, {len(self.xys)} features)"
