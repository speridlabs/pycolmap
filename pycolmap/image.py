import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray

from .utils import qvec2rotmat
from .types import INVALID_POINT3D_ID


class Image:
    """
    Represents image extrinsic parameters and features.
    NOTE: In the optimized version, instances of this class are typically created
    on-demand for accessing data and are not stored persistently in bulk.
    Features (`xys`, `point3D_ids`) are provided as NumPy arrays.
    """

    id: int
    name: str
    camera_id: int

    xys: NDArray[np.float64] # Shape: (N, 2), dtype=float64
    point3D_ids: NDArray[np.uint32] # Shape: (N,), dtype=int64

    qvec: NDArray[np.float64] # Shape: (4,), dtype=float64 [w, x, y, z]
    tvec: NDArray[np.float64] # Shape: (3,), dtype=float64 [x, y, z]

    # TODO: add optional here
    def __init__(self, id: int, name: str, camera_id: int,
                 qvec: NDArray[np.float64], tvec: NDArray[np.float64],
                 xys: NDArray[np.float64], point3D_ids: NDArray[np.uint32]):
        """
        Initializes an Image instance. Typically called internally by view objects.

        Args:
            id: Unique image identifier.
            name: Image file name.
            camera_id: ID of the camera used for this image.
            qvec: NumPy array (4,) quaternion rotation [w, x, y, z].
            tvec: NumPy array (3,) translation vector [x, y, z].
            xys: NumPy array (N, 2) of 2D feature points.
            point3D_ids: NumPy array (N,) of corresponding 3D point IDs.
        """
        if qvec.shape != (4,) or tvec.shape != (3,):
             raise ValueError("qvec must have shape (4,) and tvec shape (3,)")
        if xys.ndim != 2 or xys.shape[1] != 2:
             raise ValueError("xys must be an Nx2 array")
        if point3D_ids.ndim != 1 or point3D_ids.shape[0] != xys.shape[0]:
            raise ValueError(f"Number of 2D points ({xys.shape[0]}) does not match number of 3D point IDs ({point3D_ids.shape[0]})")

        self.id = id
        self.name = name
        self.camera_id = camera_id
        self.qvec = qvec.astype(np.float64)
        self.tvec = tvec.astype(np.float64)
        self.xys = xys.astype(np.float64)
        self.point3D_ids = point3D_ids.astype(np.uint32) # Use standard int64

    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from quaternion."""
        # qvec is already a numpy array
        return qvec2rotmat(self.qvec)

    def get_world_to_camera_matrix(self) -> np.ndarray:
        """Get world-to-camera transformation matrix.
        
        Returns:
            4x4 transformation matrix
        """

        R = self.get_rotation_matrix()
        t = self.tvec.reshape(3, 1)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R
        transform[:3, 3] = t.ravel()

        return transform

    def get_camera_to_world_matrix(self) -> np.ndarray:
        """Get camera-to-world transformation matrix.
        
        Returns:
            4x4 transformation matrix
        """

        # TODO: Avoid recalculating W2C if possible, but this is simplest for now
        world_to_camera = self.get_world_to_camera_matrix()

        # Use efficient inversion properties: C2W = [R.T | -R.T @ t]
        R_T = world_to_camera[:3, :3].T
        neg_R_T_t = -R_T @ world_to_camera[:3, 3]

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R_T
        transform[:3, 3] = neg_R_T_t

        # return np.linalg.inv(world_to_camera) # Slower alternative
        return transform

    def get_camera_center(self) -> np.ndarray:
        """Get camera center in world coordinates.
        
        Returns:
            Camera center as (x, y, z)
        """
        # C = -R' * t
        R = self.get_rotation_matrix()
        t = self.tvec
        center = -R.T @ t

        return center

    def num_observations(self) -> int:
        """Counts the number of 2D features in this image."""
        return self.xys.shape[0]

    def num_valid_observations(self) -> int:
        """Counts the number of 2D features with valid 3D correspondences."""
        return np.sum(self.point3D_ids != INVALID_POINT3D_ID)

    def get_valid_points3D(self) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Returns a list of tuples containing (point3D_id, xy_coords) for features
        that have a valid 3D point correspondence.

        Returns:
            A list of (point3D_id, (x, y)) tuples.
            NOTE: Returns list of tuples for API compatibility, though internally uses arrays.
        """
        valid_mask = self.point3D_ids != INVALID_POINT3D_ID
        valid_ids = self.point3D_ids[valid_mask]
        valid_xys = self.xys[valid_mask]
        # Convert back to list of tuples for external API consistency
        return [(int(p3d_id), tuple(xy)) for p3d_id, xy in zip(valid_ids, valid_xys)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented

        # Check shapes first for arrays
        if self.xys.shape != other.xys.shape or self.point3D_ids.shape != other.point3D_ids.shape:
             return False

        return self.id == other.id and \
               self.name == other.name and \
               self.camera_id == other.camera_id and \
               np.allclose(self.qvec, other.qvec) and \
               np.allclose(self.tvec, other.tvec) and \
               np.allclose(self.xys, other.xys) and \
               np.array_equal(self.point3D_ids, other.point3D_ids)

    def __hash__(self) -> int:
        # Hash based on ID and name primarily. Pose/features might change.
        return hash((self.id, self.name))

    def __repr__(self) -> str:
        num_feats = self.num_observations()
        return f"Image(id={self.id}, name='{self.name}', camera_id={self.camera_id}, {num_feats} features)"
