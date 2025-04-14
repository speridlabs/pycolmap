from typing import List, Tuple, Optional


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

    def remove_observation(self, image_id: int, point2D_idx: Optional[int] = None) -> bool:
        """
        Removes observations corresponding to a given image ID from the track.
        If point2D_idx is specified, only that specific observation is removed.

        Args:
            image_id: The ID of the image whose observation(s) should be removed.
            point2D_idx: If provided, remove only the observation matching this
                         specific image_id and point2D_idx pair. Otherwise, remove
                         all observations for the given image_id.

        Returns:
            True if at least one observation was removed, False otherwise.
        """
        removed = False
        new_image_ids = []
        new_point2D_idxs = []

        for i, current_img_id in enumerate(self.image_ids):
            current_p2d_idx = self.point2D_idxs[i]
            should_remove = (current_img_id == image_id) and \
                            (point2D_idx is None or current_p2d_idx == point2D_idx)
            if not should_remove:
                new_image_ids.append(current_img_id)
                new_point2D_idxs.append(current_p2d_idx)
            else:
                removed = True

        if removed:
            self.image_ids = new_image_ids
            self.point2D_idxs = new_point2D_idxs

        return removed

    def update_track(self, image_ids: List[int], point2D_idxs: List[int]) -> None:
        """
        Updates the observation track for this point.

        Args:
            image_ids: New list of observing image IDs.
            point2D_idxs: New list of corresponding 2D point indices.

        Raises:
            ValueError: If the lengths of `image_ids` and `point2D_idxs` do not match.
        """
        if len(image_ids) != len(point2D_idxs):
            raise ValueError(
                 f"Point3D {self.id}: Update failed. Number of image IDs ({len(image_ids)}) "
                 f"must match number of point2D indices ({len(point2D_idxs)})."
             )
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs

    def get_track(self) -> List[Tuple[int, int]]:
        """
        Returns the observation track as a list of (image_id, point2D_idx) pairs.

        Returns:
            A list of tuples, where each tuple contains an observing image ID
            and the index of the corresponding 2D feature in that image.
        """
        return list(zip(self.image_ids, self.point2D_idxs))
    
    def get_track_length(self) -> int:
        """Get the number of images that observe this point.
        
        Returns:
            Number of observations
        """
        return len(self.image_ids)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point3D):
            return NotImplemented

        return self.id == other.id and \
               self.xyz == other.xyz and \
               self.rgb == other.rgb and \
               self.error == other.error and \
               self.image_ids == other.image_ids and \
               self.point2D_idxs == other.point2D_idxs

    def __hash__(self) -> int:
        # Primarily hash by ID for dictionary keys, etc.
        return hash(self.id)
        
    def __repr__(self) -> str:
        return f"Point3D(id={self.id}, xyz={self.xyz}, track_length={self.get_track_length()}, error={self.error:.2f})"
