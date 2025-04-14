from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from .image import Image
from .camera import Camera
from .point3d import Point3D

from .types import INVALID_POINT3D_ID 
from .io import read_model, write_model
from .utils import find_model_path, angle_between_rays

class ColmapReconstruction:

    """
    Manages a COLMAP reconstruction, providing an API for accessing,
    manipulating, and saving camera parameters, image poses, features,
    and 3D points.

    Attributes:
        path (str): The path to the directory where the model (cameras, images, points3D) was loaded from or last saved to.
        cameras (Dict[int, Camera]): Dictionary mapping camera IDs to Camera objects.
        images (Dict[int, Image]): Dictionary mapping image IDs to Image objects.
        points3D (Dict[int, Point3D]): Dictionary mapping 3D point IDs to Point3D objects.
    """

    path: str
    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    points3D: Dict[int, Point3D]

    # Internal lookups and state for efficiency and ID management
    _image_name_to_id: Dict[str, int]
    _last_camera_id: int
    _last_image_id: int
    _last_point3D_id: int

    def __init__(self, reconstruction_path: str):
        """
        Loads a COLMAP reconstruction from a specified path.

        Searches for the model files (cameras, images, points3D) in standard
        locations ('sparse/0', 'sparse', root) within the `reconstruction_path`.
        Automatically detects binary or text format.

        Args:
            reconstruction_path: The path to the COLMAP project directory
                                 (e.g., '/path/to/project').

        Raises:
            FileNotFoundError: If no valid COLMAP model directory can be found
                             within the specified path.
            ValueError: If the model format cannot be determined or if there's
                        an issue during loading.
        """
        model_dir = find_model_path(reconstruction_path)

        if model_dir is None:
            raise FileNotFoundError(
                f"Could not find COLMAP model files in standard locations "
                f"(sparse/0, sparse, or root) within '{reconstruction_path}'"
            )

        try:
            values = read_model(model_dir)
        except (FileNotFoundError, ValueError, EOFError, RuntimeError) as e:
            raise ValueError(f"Failed to load COLMAP model from '{model_dir}': {e}")

        self.path = model_dir
        self.cameras, self.images, self.points3D = values

        self._image_name_to_id = {img.name: img_id for img_id, img in self.images.items()}
        self._last_camera_id = max(self.cameras.keys()) if self.cameras else 0
        self._last_image_id = max(self.images.keys()) if self.images else 0
        self._last_point3D_id = max(self.points3D.keys()) if self.points3D else 0

        # Verify consistency between images and points3D (optional but recommended)
        # TODO: eliminate this
        self._verify_consistency() # Can be slow for large models

    def save(self, output_path: Optional[str] = None, binary: bool = True) -> None:
        """
        Saves the current state of the reconstruction to disk.

        Args:
            output_path: The directory to save the model files (cameras, images,
                         points3D) into. If None, saves back to the original
                         load path (`self.path`). The directory will be created
                         if it doesn't exist. Defaults to None.
            binary: If True, saves in binary format (.bin). If False, saves in
                    text format (.txt). Defaults to True.

        Raises:
            RuntimeError: If saving fails for any reason.
        """
        save_dir = output_path if output_path is not None else self.path

        try:
            write_model(self.cameras, self.images, self.points3D, save_dir, binary)
            if output_path is not None: self.path = save_dir
        except Exception as e:
            raise RuntimeError(f"Failed to save reconstruction to '{save_dir}': {e}")

    def get_image_by_name(self, name: str) -> Optional[Image]:
        """
        Retrieves an Image object by its filename.
        """
        image_id = self._image_name_to_id.get(name)
        return self.images.get(image_id) if image_id is not None else None


    def add_camera(self, model: str, width: int, height: int, params: List[float]) -> int:
        """
        Adds a new camera to the reconstruction.
        Assigns a new unique camera ID.

        Args:
            model: The COLMAP camera model name (e.g., "PINHOLE").
            width: Image width in pixels.
            height: Image height in pixels.
            params: List of intrinsic parameters matching the model.

        Returns:
            The newly assigned camera ID.

        Raises:
            ValueError: If the model name is invalid or parameters are incorrect.
        """
        self._last_camera_id += 1
        new_id = self._last_camera_id

        try:
            new_camera = Camera(id=new_id, model=model, width=width, height=height, params=params)
            self.cameras[new_id] = new_camera
            return new_id
        except ValueError as e:
            self._last_camera_id -= 1
            raise e

    def add_image(self, name: str, camera_id: int,
                  qvec: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
                  tvec: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> int:
        """
        Adds a new, initially featureless, image to the reconstruction.

        Assigns a new unique image ID if `image_id` is not provided. If an ID
        is provided, it checks for collisions.

        Args:
            name: The filename of the image. Must be unique.
            camera_id: The ID of the camera used for this image. Must exist.
            qvec: Rotation quaternion (w, x, y, z). Defaults to identity.
            tvec: Translation vector (x, y, z). Defaults to origin.
            image_id: Optional specific ID to assign. If None, a new ID is generated.

        Returns:
            The assigned image ID (either provided or newly generated).

        Raises:
            ValueError: If the name is already used, the camera ID doesn't exist,
                        or the provided image_id is already taken.
        """

        if name in self._image_name_to_id:
            raise ValueError(f"Image name '{name}' already exists (ID: {self._image_name_to_id[name]}).")

        if camera_id not in self.cameras:
            raise ValueError(f"Camera with ID {camera_id} does not exist.")

        self._last_image_id += 1
        new_id = self._last_image_id

        try:
            new_image = Image(id=new_id, name=name, camera_id=camera_id, qvec=qvec, tvec=tvec)
            self.images[new_id] = new_image
            self._image_name_to_id[name] = new_id
        except ValueError as e:
            self._last_image_id -= 1
            raise e

        return new_id


    def add_point3D(self, xyz: Tuple[float, float, float],
                    track: List[Tuple[int, int]], # List of (image_id, point2d_idx)
                    rgb: Tuple[int, int, int] = (128, 128, 128),
                    error: float = 0.0) -> int:
        """
        Adds a new 3D point to the reconstruction.

        Assigns a new unique point ID if `point3D_id` is not provided. Updates
        the `point3D_ids` list in the corresponding Image objects based on the track.

        Args:
            xyz: The 3D coordinates (x, y, z) of the point.
            track: A list of tuples `(image_id, point2D_idx)` specifying which 2D
                   features in which images observe this point.
            rgb: The color (r, g, b). Defaults to gray.
            error: The reprojection error. Defaults to 0.0.

        Returns:
            The assigned 3D point ID.

        Raises:
            ValueError: If validation fails (e.g., provided ID exists, track is invalid,
                        an image_id in the track doesn't exist, or a point2D_idx is out
                        of bounds or already assigned to another 3D point).
        """
        self._last_point3D_id += 1
        new_id = self._last_point3D_id
        image_ids = [item[0] for item in track]
        point2D_idxs = [item[1] for item in track]
        images_to_update: Dict[int, List[int]] = defaultdict(list) # image_id -> list of point2D_idx to update

        def check_point_collision(img_id: int, p2d_idx: int) -> str:
            if img_id not in self.images:
                return f"Cannot add point {new_id}: Image ID {img_id} in track does not exist."
            image = self.images[img_id]
            if not (0 <= p2d_idx < len(image.xys)):
                return f"Cannot add point {new_id}: Point2D index {p2d_idx} is out of bounds for image {img_id} (size: {len(self.images[img_id].xys)})."
            if image.point3D_ids[p2d_idx] != INVALID_POINT3D_ID:
                if image.point3D_ids[p2d_idx] != new_id:
                    return f"Cannot add point {new_id}: 2D feature at index {p2d_idx} in image {img_id} is already assigned to Point3D ID {image.point3D_ids[p2d_idx]}."
            return ""

        for img_id, p2d_idx in track:
            error_msg: str = check_point_collision(img_id, p2d_idx)
            if error_msg != "": raise ValueError(error_msg)
            images_to_update[img_id].append(p2d_idx)

        try:
            new_point = Point3D(
                id=new_id, 
                xyz=xyz, 
                rgb=rgb, 
                error=error,
                image_ids=image_ids, 
                point2D_idxs=point2D_idxs
            )
        except ValueError as e:
            self._last_point3D_id -= 1
            raise e

        self.points3D[new_id] = new_point

        # Update the corresponding images' point3D_ids list
        try:
            for img_id, indices in images_to_update.items():
                 for p2d_idx in indices:
                    if 0 <= p2d_idx < len(self.images[img_id].point3D_ids):
                        self.images[img_id].point3D_ids[p2d_idx] = new_id

        except Exception as e:
            del self.points3D[new_id]
            self._last_point3D_id -= 1
            # TODO: Reverting image changes is complex, might need temporary storage or Image methods
            raise RuntimeError(f"Failed to update image tracks for Point3D {new_id}.") from e

        return new_id

    def delete_camera(self, camera_id: int, force: bool = False) -> None:
        """
        Deletes a camera from the reconstruction.

        Args:
            camera_id: The ID of the camera to delete.
            force: If True, also deletes all images using this camera. If False
                   (default), raises an error if the camera is still in use.

        Returns:
            True if the camera was deleted, False if it was not found.

        Raises:
            ValueError: If `force` is False and the camera is still used by images.
        """
        if camera_id not in self.cameras: return
        images_using_camera = [img_id for img_id, img in self.images.items() if img.camera_id == camera_id]

        if len(images_using_camera) > 0:
            if not force:
                raise ValueError(
                    f"Camera {camera_id} is used by {len(images_using_camera)} images "
                    f"(e.g., ID {images_using_camera[0]}). Use force=True to delete images too."
                )

            for img_id in list(images_using_camera):
                self.delete_image(img_id)

        del self.cameras[camera_id]

    def delete_image(self, image_id: int) -> None:
        """
        Deletes an image from the reconstruction.

        Also removes this image's observations from the tracks of all 3D points
        it observes. If removing an observation causes a 3D point's track to
        become empty, that 3D point is also deleted.

        Args:
            image_id: The ID of the image to delete.

        Returns:
            True if the image was deleted, False if it was not found.
        """
        if image_id not in self.images: return 
        image_to_delete = self.images[image_id]
        points_to_check_for_deletion: Set[int] = set()

        # Removes point observations from image
        for p2d_idx, point3D_id in enumerate(image_to_delete.point3D_ids):
            if point3D_id != INVALID_POINT3D_ID and point3D_id in self.points3D:
                point = self.points3D[point3D_id]
                removed = point.remove_observation(image_id, p2d_idx)
                if removed and point.get_track_length() == 0:
                    points_to_check_for_deletion.add(point3D_id)

        # Delete from scene points whose tracks became empty
        for point3D_id in points_to_check_for_deletion:
            if point3D_id in self.points3D and self.points3D[point3D_id].get_track_length() == 0:
                del self.points3D[point3D_id]

        # Removes image from internal lookup
        if image_to_delete.name in self._image_name_to_id:
            del self._image_name_to_id[image_to_delete.name]

        del self.images[image_id]


    def delete_point3D(self, point3D_id: int) -> None:
        """
        Deletes a 3D point from the reconstruction.

        Also updates all images that observed this point, setting the corresponding
        `point3D_ids` entry in their feature list to `INVALID_POINT3D_ID`.

        Args:
            point3D_id: The ID of the 3D point to delete.

        Returns:
            True if the point was deleted, False if it was not found.
        """
        if point3D_id not in self.points3D: return

        point_to_delete = self.points3D[point3D_id]

        # Update observing images to invalidate their reference to this point
        for image_id, p2d_idx in point_to_delete.get_track():
            if image_id not in self.images:
                print(f"Warning: Inconsistent track data for Point3D {point3D_id}. "
                          f"Image {image_id} has no point2D at index {p2d_idx}.")
                continue

            image = self.images[image_id]

            # Check bounds just in case track data is inconsistent
            if 0 <= p2d_idx < len(image.point3D_ids):
                    # Assuming Image features can be modified directly or via method
                p3d_ids_list = list(image.point3D_ids) # Create mutable copy if needed
                # Only invalidate if it currently points to the point we're deleting
                if p3d_ids_list[p2d_idx] == point3D_id:
                    p3d_ids_list[p2d_idx] = INVALID_POINT3D_ID
                    # Update the image object
                    image.update_features(image.xys, p3d_ids_list)

        del self.points3D[point3D_id]

    def filter_points3D(self,
                       min_track_len: int = 2,
                       max_error: float = float('inf'),
                       min_angle: float = 0.0) -> int:
        """
        Filters the 3D point cloud based on track length, reprojection error,
        and minimum triangulation angle criteria. Points failing any criterion
        are deleted.

        Args:
            min_track_len: Minimum number of images observing a point. Defaults to 2.
            max_error: Maximum allowed mean reprojection error. Defaults to infinity.
            min_angle: Minimum triangulation angle (in degrees) between any two
                       observing camera rays. Ignored if < 2 observations or <= 0.
                       Defaults to 0.0.

        Returns:
            The number of 3D points that were deleted.
        """
        points_to_delete: Set[int] = set()

        for point3D_id, point in self.points3D.items():
            # Criterion 1: Track Length
            track_len = point.get_track_length()
            if track_len < min_track_len:
                points_to_delete.add(point3D_id)
                continue

            # Criterion 2: Reprojection Error
            if point.error > max_error:
                points_to_delete.add(point3D_id)
                continue

            # Criterion 3: Triangulation Angle (only if needed and possible)
            if min_angle > 0.0 and track_len >= 2:
                observing_centers = []
                for img_id, _ in point.get_track():
                    if img_id in self.images:
                        # Use cached camera center for efficiency
                        observing_centers.append(self.images[img_id].get_camera_center())
                    # else: image might have been deleted, skip this observation for angle calc

                # Need at least two valid camera centers to calculate angles
                if len(observing_centers) >= 2:
                    max_observed_angle = 0.0
                    # Check angle between all pairs of valid observing cameras
                    for i in range(len(observing_centers)):
                        for j in range(i + 1, len(observing_centers)):
                            angle = angle_between_rays(
                                observing_centers[i], observing_centers[j], point.xyz
                            )
                            max_observed_angle = max(max_observed_angle, angle)

                    # If the largest angle found is still too small, filter the point
                    if max_observed_angle < min_angle:
                        points_to_delete.add(point3D_id)
                        continue # Skip to next point

        # Perform deletions
        deleted_count = 0
        for point_id in points_to_delete:
            if self.delete_point3D(point_id):
                deleted_count += 1

        return deleted_count

    def get_statistics(self) -> Dict[str, float]:
         """
         Calculates basic statistics about the reconstruction.

         Returns:
             A dictionary containing:
             - 'num_cameras': Number of cameras.
             - 'num_images': Number of registered images.
             - 'num_points3D': Number of 3D points.
             - 'mean_track_length': Average number of observations per 3D point.
             - 'mean_observations_per_image': Average number of 2D features per image.
             - 'mean_valid_observations_per_image': Average number of 2D features with
                                                    valid 3D correspondences per image.
             - 'mean_reprojection_error': Average reprojection error across all 3D points.
         """
         num_cameras = len(self.cameras)
         num_images = len(self.images)
         num_points = len(self.points3D)

         if num_points == 0:
             mean_track_length = 0.0
             mean_reprojection_error = 0.0
         else:
             total_track_length = sum(p.get_track_length() for p in self.points3D.values())
             mean_track_length = total_track_length / num_points
             total_error = sum(p.error for p in self.points3D.values())
             mean_reprojection_error = total_error / num_points

         if num_images == 0:
              mean_observations = 0.0
              mean_valid_observations = 0.0
         else:
             total_obs = sum(img.num_observations() for img in self.images.values())
             mean_observations = total_obs / num_images
             total_valid_obs = sum(img.num_valid_observations() for img in self.images.values())
             mean_valid_observations = total_valid_obs / num_images


         return {
             "num_cameras": float(num_cameras),
             "num_images": float(num_images),
             "num_points3D": float(num_points),
             "mean_track_length": mean_track_length,
             "mean_observations_per_image": mean_observations,
             "mean_valid_observations_per_image": mean_valid_observations,
             "mean_reprojection_error": mean_reprojection_error,
         }


    def __str__(self) -> str:
        """Provides a string summary of the reconstruction."""
        stats = self.get_statistics()
        return (
            f"ColmapReconstruction(path='{self.path}', "
            f"cameras={int(stats['num_cameras'])}, images={int(stats['num_images'])}, "
            f"points3D={int(stats['num_points3D'])}, "
            f"mean_track_len={stats['mean_track_length']:.2f}, "
            f"mean_reproj_err={stats['mean_reprojection_error']:.2f})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    # --- Consistency Checks (Optional) ---
    def _verify_consistency(self) -> None:
        """
        Performs internal consistency checks. Useful for debugging.
        Can be slow on large models.
        """

        print("Performing consistency checks...")

        # Check image camera IDs exist
        for img_id, img in self.images.items():
            if img.camera_id not in self.cameras:
                print(f"Error: Image {img_id} uses non-existent camera ID {img.camera_id}")

        # Check point tracks reference valid images and indices
        point_obs_counts = defaultdict(int)
        for p3d_id, p3d in self.points3D.items():
            if len(p3d.image_ids) != len(p3d.point2D_idxs):
                 print(f"Error: Point3D {p3d_id} has mismatched track lengths ({len(p3d.image_ids)} vs {len(p3d.point2D_idxs)})")
            for i, (img_id, p2d_idx) in enumerate(p3d.get_track()):
                 point_obs_counts[p3d_id] += 1
                 if img_id not in self.images:
                     print(f"Error: Point3D {p3d_id} track references non-existent image ID {img_id}")
                 else:
                     image = self.images[img_id]
                     if not (0 <= p2d_idx < len(image.xys)):
                          print(f"Error: Point3D {p3d_id} track references out-of-bounds point2D index {p2d_idx} for image {img_id} (size {len(image.xys)})")
                     # Check back-reference from image
                     elif image.point3D_ids[p2d_idx] != p3d_id:
                          print(f"Error: Point3D {p3d_id} track inconsistency. Image {img_id} point2D index {p2d_idx} points to {image.point3D_ids[p2d_idx]} instead.")

        # Check image features reference valid points
        image_obs_counts = defaultdict(int)
        for img_id, img in self.images.items():
            if len(img.xys) != len(img.point3D_ids):
                print(f"Error: Image {img_id} has mismatched feature lengths ({len(img.xys)} vs {len(img.point3D_ids)})")
            for p2d_idx, p3d_id in enumerate(img.point3D_ids):
                if p3d_id != INVALID_POINT3D_ID:
                     image_obs_counts[img_id] += 1
                     if p3d_id not in self.points3D:
                         print(f"Error: Image {img_id} point2D index {p2d_idx} references non-existent Point3D ID {p3d_id}")

        print("Consistency checks finished.")
