import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Union

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


    def __init__(self, id: int,
                 xyz: Union[NDArray[np.float64], Tuple[float, float, float]],
                 rgb: Union[NDArray[np.uint8], Tuple[int, int, int]],
                 error: float,
                 image_ids: Union[NDArray[np.int64], List[int]],
                 point2D_idxs: Union[NDArray[np.int64], List[int]]):
        """
        Initialize a 3D point. Typically called internally by view objects.

        Args:
            id: Unique point identifier.
            xyz: NumPy array (3,) or tuple (3,) 3D coordinates [x, y, z].
            rgb: NumPy array (3,) or tuple (3,) RGB color [r, g, b] (0-255).
            error: Reprojection error.
            image_ids: NumPy array (T,) or list (T,) of image IDs where this point is visible.
            point2D_idxs: NumPy array (T,) or list (T,) of corresponding 2D point indices.
        """
        # Convert inputs to numpy arrays first
        xyz_arr = np.array(xyz, dtype=np.float64)
        rgb_arr = np.array(rgb, dtype=np.uint8)
        image_ids_arr = np.array(image_ids, dtype=np.int64)
        point2D_idxs_arr = np.array(point2D_idxs, dtype=np.int64)

        # Perform checks on the converted numpy arrays
        if xyz_arr.shape != (3,): raise ValueError("xyz must have shape (3,)")
        if rgb_arr.shape != (3,): raise ValueError("rgb must have shape (3,)")
        if image_ids_arr.ndim != 1 or point2D_idxs_arr.ndim != 1:
             if not ((image_ids_arr.ndim == 1 and image_ids_arr.shape[0] == 0) and \
                     (point2D_idxs_arr.ndim == 1 and point2D_idxs_arr.shape[0] == 0)):
                 raise ValueError("image_ids and point2D_idxs must be 1D arrays or empty lists.")
        if image_ids_arr.shape != point2D_idxs_arr.shape:
            if not (image_ids_arr.shape[0] == 0 and point2D_idxs_arr.shape[0] == 0):
                raise ValueError(f"Number of image IDs ({image_ids_arr.shape[0]}) does not match number of point2D indices ({point2D_idxs_arr.shape[0]})")

        self.id = id
        self.xyz = xyz_arr
        self.rgb = rgb_arr
        self.error = float(error)
        self.image_ids = image_ids_arr
        self.point2D_idxs = point2D_idxs_arr


    def get_track(self) -> List[Tuple[int, int]]:
        """
        Returns the observation track as a list of (image_id, point2D_idx) pairs.
        NOTE: Returns list of tuples for API compatibility.
        """
        # Convert back to list of tuples for external API consistency
        return [(int(img_id), int(p2d_idx))
                for img_id, p2d_idx in zip(self.image_ids, self.point2D_idxs)]

    def get_track_length(self) -> int:
        """Get the number of images that observe this point."""
        return len(self.image_ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point3D):
            return NotImplemented

        if self.image_ids.shape != other.image_ids.shape or self.point2D_idxs.shape != other.point2D_idxs.shape:
             return False

        # Use np.isclose for float comparisons (error, xyz)
        return bool(self.id == other.id and \
                    np.allclose(self.xyz, other.xyz) and \
                    np.array_equal(self.rgb, other.rgb) and \
                    np.isclose(self.error, other.error) and \
                    np.array_equal(self.image_ids, other.image_ids) and \
                    np.array_equal(self.point2D_idxs, other.point2D_idxs))

    def __hash__(self) -> int:
        # Primarily hash by ID for dictionary keys, etc.
        return hash(self.id)

    def __repr__(self) -> str:
        xyz_str = np.array2string(self.xyz, precision=3, separator=', ', suppress_small=True)
        return f"Point3D(id={self.id}, xyz={xyz_str}, track_length={self.get_track_length()}, error={self.error:.2f})"

