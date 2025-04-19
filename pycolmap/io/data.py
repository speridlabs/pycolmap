import numpy as np
from typing import List
from numpy.typing import NDArray

class CameraData:
    __slots__ = ['num_cameras', 'ids', 'model_ids', 'widths', 'heights', 'params']
    
    num_cameras: int
    ids: NDArray[np.uint16]
    model_ids: NDArray[np.uint8]
    widths: NDArray[np.uint32]
    heights: NDArray[np.uint32]
    params: NDArray[np.float32]  # Shape (N, MAX_CAMERA_PARAMS)
    
    def __init__(self):
        self.num_cameras = 0


class ImageData:
    __slots__ = ['names', 'num_images', 'ids', 'camera_ids', 'qvecs', 
                 'tvecs', 'all_xys', 'all_point3D_ids', 'feature_indices']
    
    names: List[str]
    num_images: int
    ids: NDArray[np.uint16]
    camera_ids: NDArray[np.uint16]
    qvecs: NDArray[np.float64]  # Shape (N, 4)
    tvecs: NDArray[np.float64]  # Shape (N, 3)
    all_xys: NDArray[np.float64]  # Shape (TotalPoints, 2)
    all_point3D_ids: NDArray[np.uint32]  # Shape (TotalPoints,)
    feature_indices: NDArray[np.uint32]  # Shape (N, 2)
    
    def __init__(self):
        self.num_images = 0
        self.names = []


class Point3DData:
    __slots__ = ['num_points', 'ids', 'rgbs', 'xyzs', 'errors', 
                 'all_track_image_ids', 'all_track_point2D_idxs', 'track_indices']
    
    num_points: int
    ids: NDArray[np.uint32]
    rgbs: NDArray[np.uint8]  # Shape (M, 3)
    xyzs: NDArray[np.float64]  # Shape (M, 3)
    errors: NDArray[np.float64]  # Shape (M,)
    all_track_image_ids: NDArray[np.uint16]  # Shape (TotalTrackLen,)
    all_track_point2D_idxs: NDArray[np.uint32]  # Shape (TotalTrackLen,)
    track_indices: NDArray[np.uint32]  # Shape (M, 2)
    
    def __init__(self):
        self.num_points = 0

