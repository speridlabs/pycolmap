import numpy as np
from typing import List
from numpy.typing import NDArray

# --- Internal Data Structures for IO Results ---
# These hold data temporarily before being consolidated in ColmapReconstruction

class CameraData:
    num_cameras: int

    ids: NDArray[np.uint16]
    model_ids: NDArray[np.uint8]

    widths: NDArray[np.uint32]
    heights: NDArray[np.uint32]

    # Shape (N, MAX_CAMERA_PARAMS), padded with NaNs
    params: NDArray[np.float32] # (N, MAX_CAMERA_PARAMS)
    "Shape [?, MAX_CAMERA_PARAMS]"

class ImageData:
    names: List[str]
    num_images: int

    ids: NDArray[np.uint16]
    camera_ids: NDArray[np.uint16]

    qvecs: NDArray[np.float64] # (N, 4)
    "Shape [?, 4]"
    tvecs: NDArray[np.float64] # (N, 3)
    "Shape [?, 3]"

    # Features stored concatenated
    all_xys: NDArray[np.float64] # (TotalPoints, 2)
    "Shape [?, 2]"
    all_point3D_ids: NDArray[np.uint32] # (TotalPoints,)
    # Indices into all_xys and all_point3D_ids for each image [start, end)
    feature_indices: NDArray[np.uint32] # (N, 2)
    "Shape [?, 2]"


class Point3DData:
    num_points: int

    ids: NDArray[np.uint32]

    rgbs: NDArray[np.uint8] # (M, 3)
    "Shape [?, 3]"
    xyzs: NDArray[np.float64] # (M, 3)
    "Shape [?, 3]"
    errors: NDArray[np.float64] # (M,)

    # Tracks stored concatenated
    all_track_image_ids: NDArray[np.uint16] # (TotalTrackLen,)
    all_track_point2D_idxs: NDArray[np.uint32] # (TotalTrackLen,)
    # Indices into all_track arrays for each point [start, end)
    track_indices: NDArray[np.uint32] # (M, 2)
    "Shape [?, 2]"
