import numpy as np
from typing import Tuple, List 
from numpy.typing import NDArray

class Point3D:
    """
    Represents a 3D point with color and track information.
    NOTE: In the optimized version, instances of this class are typically created
    on-demand for accessing data and are not stored persistently in bulk.
    Track information (`image_ids`, `point2D_idxs`) is provided as NumPy arrays.
    """

    id: int
    error: float  # Reprojection error

    # Provided as numpy arrays when instantiated by the reconstruction view
    image_ids: NDArray[np.int64]    # Shape: (T,)
    point2D_idxs: NDArray[np.int64] # Shape: (T,)

    rgb: NDArray[np.uint8]          # Shape: (3,) [r, g, b]
    xyz: NDArray[np.float64]        # Shape: (3,) [x, y, z]


    def __init__(self, id: int, xyz: np.ndarray, rgb: np.ndarray,
                 error: float, image_ids: np.ndarray, point2D_idxs: np.ndarray):
        """
        Initialize a 3D point. Typically called internally by view objects.

        Args:
            id: Unique point identifier.
            xyz: NumPy array (3,) 3D coordinates [x, y, z].
            rgb: NumPy array (3,) RGB color [r, g, b].
            error: Reprojection error.
            image_ids: NumPy array (T,) List of image IDs where this point is visible.
            point2D_idxs: NumPy array (T,) List of corresponding 2D point indices.
        """
        if xyz.shape != (3,): raise ValueError("xyz must have shape (3,)")
        if rgb.shape != (3,): raise ValueError("rgb must have shape (3,)")
        if image_ids.ndim != 1 or point2D_idxs.ndim != 1:
             raise ValueError("image_ids and point2D_idxs must be 1D arrays.")
        if image_ids.shape != point2D_idxs.shape:
            raise ValueError(f"Number of image IDs ({len(image_ids)}) does not match number of point2D indices ({len(point2D_idxs)})")

        self.id = id
        self.xyz = xyz.astype(np.float64)
        self.rgb = rgb.astype(np.uint8)
        self.error = float(error)
        self.image_ids = image_ids.astype(np.int64)
        self.point2D_idxs = point2D_idxs.astype(np.int64)


    def get_track(self) -> List[Tuple[int, int]]:
        """
        Returns the observation track as a list of (image_id, point2D_idx) pairs.
        NOTE: Returns list of tuples for API compatibility.
        """
        # Convert back to list of tuples for external API consistency
        return list(zip(self.image_ids.tolist(), self.point2D_idxs.tolist()))

    def get_track_length(self) -> int:
        """Get the number of images that observe this point."""
        return len(self.image_ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point3D):
            return NotImplemented

        if self.image_ids.shape != other.image_ids.shape or self.point2D_idxs.shape != other.point2D_idxs.shape:
             return False

        # Use np.isclose for float comparisons (error, xyz)
        return self.id == other.id and \
               np.allclose(self.xyz, other.xyz) and \
               np.array_equal(self.rgb, other.rgb) and \
               np.isclose(self.error, other.error) and \
               np.array_equal(self.image_ids, other.image_ids) and \
               np.array_equal(self.point2D_idxs, other.point2D_idxs)

    def __hash__(self) -> int:
        # Primarily hash by ID for dictionary keys, etc.
        return hash(self.id)

    def __repr__(self) -> str:
        xyz_str = np.array2string(self.xyz, precision=3, separator=', ', suppress_small=True)
        return f"Point3D(id={self.id}, xyz={xyz_str}, track_length={self.get_track_length()}, error={self.error:.2f})"

