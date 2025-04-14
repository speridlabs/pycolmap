from typing import List, Tuple


class Point3D:
    """3D point with color and track information."""
    
    id: int
    error: float  # Reprojection error
    image_ids: List[int] 
    point2D_idxs: List[int] 

    rgb: Tuple[int, int, int]  # [r, g, b]
    xyz: Tuple[float, float, float]  # [x, y, z]

    
    def __init__(self, id: int, xyz: Tuple[float, float, float], rgb: Tuple[int, int, int],
                 error: float, image_ids: List[int], point2D_idxs: List[int]):
        """Initialize a 3D point.
        
        Args:
            id: Unique point identifier
            xyz: 3D coordinates [x, y, z]
            rgb: RGB color [r, g, b]
            error: Reprojection error
            image_ids: List of image IDs where this point is visible
            point2D_idxs: List of corresponding 2D point indices in each image
        """
        self.id = id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs
        
        if len(image_ids) != len(point2D_idxs):
            raise ValueError(f"Number of image IDs ({len(image_ids)}) does not match number of point2D indices ({len(point2D_idxs)})")
    
    def get_track_length(self) -> int:
        """Get the number of images that observe this point.
        
        Returns:
            Number of observations
        """
        return len(self.image_ids)
        
    def __repr__(self) -> str:
        return f"Point3D(id={self.id}, xyz={self.xyz}, track_length={self.get_track_length()}, error={self.error:.2f})"
