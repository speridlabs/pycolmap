import concurrent.futures
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from .image import Image
from .camera import Camera
from .point3d import Point3D

from .types import INVALID_POINT3D_ID 
from .utils import find_model_path, angle_between_rays
from .io import read_model, write_model, InternalReconstructionData

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

    def __init__(self, reconstruction_path: str, verify_integrity:bool = True, only_3d_features: bool = True) -> None:
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
            values = read_model(model_dir, only_3d_features=only_3d_features)
        except (FileNotFoundError, ValueError, EOFError, RuntimeError) as e:
            raise ValueError(f"Failed to load COLMAP model from '{model_dir}': {e}")

        self.path = model_dir
        self.cameras, self.images, self.points3D = values

        self._image_name_to_id = {img.name: img_id for img_id, img in self.images.items()}
        self._last_camera_id = max(self.cameras.keys()) if self.cameras else 0
        self._last_image_id = max(self.images.keys()) if self.images else 0
        self._last_point3D_id = max(self.points3D.keys()) if self.points3D else 0

        if verify_integrity:
            errors = self._verify_consistency()
            if len(errors) > 0:
                print("Warning: Inconsistencies found in the reconstruction:")
                for error in errors: print(f"  - {error}")

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

    def _verify_consistency(self, max_workers: Optional[int] = 3) -> List[str]:
        """
        Performs internal consistency checks in parallel. Useful for debugging.

        Args:
            max_workers: Maximum number of threads to use. If None, defaults to
                         the number of processors on the machine, multiplied by 5,
                         considering that tasks might be I/O bound or release the GIL.

        Returns:
            List of error messages. Empty if no errors found.
        """
        all_errors: List[str] = []

        # --- Helper functions for parallel execution ---
        def _check_single_image_camera(img_item: Tuple[int, Image]) -> List[str]:
            img_id, img = img_item
            errors = []
            if img.camera_id not in self.cameras:
                errors.append(f"Image {img_id} uses non-existent camera ID {img.camera_id}")
            # Also check feature list lengths here as it's per-image
            if len(img.xys) != len(img.point3D_ids):
                 errors.append(f"Image {img_id} has mismatched feature lengths ({len(img.xys)} vs {len(img.point3D_ids)})")
            return errors

        def _check_single_point_track(p3d_item: Tuple[int, Point3D]) -> List[str]:
            p3d_id, p3d = p3d_item
            errors = []
            if len(p3d.image_ids) != len(p3d.point2D_idxs):
                errors.append(f"Point3D {p3d_id} has mismatched image and point2D index lengths ({len(p3d.image_ids)} vs {len(p3d.point2D_idxs)})")

            for img_id, p2d_idx in p3d.get_track():
                if img_id not in self.images:
                    errors.append(f"Point3D {p3d_id} track references non-existent image ID {img_id}")
                    continue # Skip further checks for this observation if image doesn't exist

                image = self.images[img_id]
                # Check bounds first
                if not (0 <= p2d_idx < len(image.xys)):
                    errors.append(f"Point3D {p3d_id} track references out-of-bounds point2D index {p2d_idx} for image {img_id} (size {len(image.xys)})")
                    continue # Skip back-reference check if index is invalid

                # Check back-reference from image
                image_p3d_id_at_idx = image.point3D_ids[p2d_idx]
                if image_p3d_id_at_idx != p3d_id:
                    errors.append(f"Point3D {p3d_id} track inconsistency. Image {img_id} point2D index {p2d_idx} points to {image_p3d_id_at_idx} instead.")
            return errors

        def _check_single_image_features(img_item: Tuple[int, Image]) -> List[str]:
            img_id, img = img_item
            errors = []
            # Length check is done in _check_single_image_camera now
            # Check if features point to valid points
            for p2d_idx, p3d_id in enumerate(img.point3D_ids):
                if p3d_id != INVALID_POINT3D_ID:
                    if p3d_id not in self.points3D:
                        errors.append(f"Image {img_id} point2D index {p2d_idx} references non-existent Point3D ID {p3d_id}")
            return errors

        # --- Execute checks in parallel ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_cam_checks = executor.map(_check_single_image_camera, self.images.items())
            future_track_checks = executor.map(_check_single_point_track, self.points3D.items())
            future_feat_checks = executor.map(_check_single_image_features, self.images.items())

            # Collect results - flatten the lists of lists
            for error_list in future_cam_checks:
                all_errors.extend(error_list)
            for error_list in future_track_checks:
                all_errors.extend(error_list)
            for error_list in future_feat_checks:
                all_errors.extend(error_list)

        return all_errors




import numpy as np
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Set, Any, Iterator, KeysView, ValuesView, ItemsView, Union

from .image import Image
from .camera import Camera
from .point3d import Point3D

from .io.binary import MAX_CAMERA_PARAMS
from .io import read_model, write_model, CameraData, ImageData, Point3DData

from .utils import find_model_path, angle_between_rays, qvec2rotmat, rotmat2qvec
from .types import INVALID_POINT3D_ID, CAMERA_MODEL_IDS, CAMERA_MODEL_NAMES

# Helper function for resizing numpy arrays (can be slow for large arrays)
def _resize_array(arr: np.ndarray, new_size: int) -> np.ndarray:
    """Resize the first dimension of a numpy array, copying data."""
    new_arr = np.zeros((new_size,) + arr.shape[1:], dtype=arr.dtype)
    copy_size = min(arr.shape[0], new_size)
    if copy_size > 0:
        new_arr[:copy_size] = arr[:copy_size]
    return new_arr


class BaseView:
    """Base class for dictionary views."""
    def __init__(self, reconstruction: 'ColmapReconstruction'):
        self._recon = reconstruction # Reference to the main reconstruction object

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, key: int) -> bool:
        raise NotImplementedError

    def __getitem__(self, key: int) -> Any:
        raise NotImplementedError

    def get(self, key: int, default: Any = None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def keys(self) -> KeysView[int]:
        raise NotImplementedError

    def values(self) -> ValuesView[Any]:
        raise NotImplementedError

    def items(self) -> ItemsView[int, Any]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        return iter(self.keys())


class CameraView(BaseView):
    """Provides dict-like access to Camera objects, created on-the-fly."""
    def __len__(self) -> int:
        return self._recon._num_cameras

    def __contains__(self, camera_id: int) -> bool:
        return camera_id in self._recon._camera_id_to_row

    def __getitem__(self, camera_id: int) -> Camera:
        if camera_id not in self._recon._camera_id_to_row:
            raise KeyError(f"Camera with ID {camera_id} not found.")
        row = self._recon._camera_id_to_row[camera_id]

        model_id = self._recon._camera_model_ids[row]
        model_name = CAMERA_MODEL_IDS[model_id].model_name
        width = self._recon._camera_widths[row]
        height = self._recon._camera_heights[row]
        params = self._recon._camera_params[row] # Already padded

        # Instantiate Camera object on-the-fly
        return Camera(id=camera_id, model=model_name, width=int(width), height=int(height), params=params)

    def keys(self) -> KeysView[int]:
        return self._recon._camera_id_to_row.keys()

    def values(self) -> ValuesView[Camera]:
        # Generator yielding on-the-fly Camera objects
        return (self[key] for key in self.keys())

    def items(self) -> ItemsView[int, Camera]:
        # Generator yielding on-the-fly (id, Camera) pairs
        return ((key, self[key]) for key in self.keys())


class ImageView(BaseView):
    """Provides dict-like access to Image objects, created on-the-fly."""
    def __len__(self) -> int:
        return self._recon._num_images

    def __contains__(self, image_id: int) -> bool:
        return image_id in self._recon._image_id_to_row

    def __getitem__(self, image_id: int) -> Image:
        if image_id not in self._recon._image_id_to_row:
            raise KeyError(f"Image with ID {image_id} not found.")
        row = self._recon._image_id_to_row[image_id]

        name = self._recon._image_names[row]
        camera_id = self._recon._image_camera_ids[row]
        qvec = self._recon._image_qvecs[row]
        tvec = self._recon._image_tvecs[row]

        # Extract features for this image
        start, end = self._recon._image_feature_indices[row]
        if start < end:
            xys = self._recon._all_xys[start:end]
            point3D_ids = self._recon._all_point3D_ids[start:end]
        else:
            xys = np.empty((0, 2), dtype=np.float64)
            point3D_ids = np.empty((0,), dtype=np.int64)

        # Instantiate Image object on-the-fly
        return Image(id=image_id, name=name, camera_id=int(camera_id),
                     qvec=qvec, tvec=tvec, xys=xys, point3D_ids=point3D_ids)

    def keys(self) -> KeysView[int]:
        return self._recon._image_id_to_row.keys()

    def values(self) -> ValuesView[Image]:
        return (self[key] for key in self.keys())

    def items(self) -> ItemsView[int, Image]:
        return ((key, self[key]) for key in self.keys())


class Point3DView(BaseView):
    """Provides dict-like access to Point3D objects, created on-the-fly."""
    def __len__(self) -> int:
        return self._recon._num_points3D

    def __contains__(self, point3D_id: int) -> bool:
        return point3D_id in self._recon._point3D_id_to_row

    def __getitem__(self, point3D_id: int) -> Point3D:
        if point3D_id not in self._recon._point3D_id_to_row:
             raise KeyError(f"Point3D with ID {point3D_id} not found.")
        row = self._recon._point3D_id_to_row[point3D_id]

        xyz = self._recon._point_xyzs[row]
        rgb = self._recon._point_rgbs[row]
        error = self._recon._point_errors[row]

        # Extract track for this point
        start, end = self._recon._point_track_indices[row]
        if start < end:
             image_ids = self._recon._all_track_image_ids[start:end]
             point2D_idxs = self._recon._all_track_point2D_idxs[start:end]
        else:
            image_ids = np.empty((0,), dtype=np.int64)
            point2D_idxs = np.empty((0,), dtype=np.int64)

        # Instantiate Point3D object on-the-fly
        return Point3D(id=point3D_id, xyz=xyz, rgb=rgb, error=float(error),
                       image_ids=image_ids, point2D_idxs=point2D_idxs)

    def keys(self) -> KeysView[int]:
        return self._recon._point3D_id_to_row.keys()

    def values(self) -> ValuesView[Point3D]:
        return (self[key] for key in self.keys())

    def items(self) -> ItemsView[int, Point3D]:
        return ((key, self[key]) for key in self.keys())


# --- Main Reconstruction Class ---

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

    # Cameras
    _camera_ids: np.ndarray
    _camera_model_ids: np.ndarray
    _camera_widths: np.ndarray
    _camera_heights: np.ndarray
    _camera_params: np.ndarray # Padded
    _camera_id_to_row: Dict[int, int]
    _num_cameras: int

    # Images
    _image_ids: np.ndarray
    _image_qvecs: np.ndarray
    _image_tvecs: np.ndarray
    _image_camera_ids: np.ndarray
    _image_names: List[str]
    _image_id_to_row: Dict[int, int]
    _image_name_to_id: Dict[str, int]
    _num_images: int

    # Image Features (Concatenated)
    _all_xys: np.ndarray
    _all_point3D_ids: np.ndarray
    _image_feature_indices: np.ndarray # (N_images, 2) [start, end)

    # 3D Points
    _point_ids: np.ndarray
    _point_xyzs: np.ndarray
    _point_rgbs: np.ndarray
    _point_errors: np.ndarray
    _point3D_id_to_row: Dict[int, int]
    _num_points3D: int

    # Point Tracks (Concatenated)
    _all_track_image_ids: np.ndarray
    _all_track_point2D_idxs: np.ndarray
    _point_track_indices: np.ndarray # (N_points, 2) [start, end)

    # --- API Facade ---
    cameras: CameraView
    images: ImageView
    points3D: Point3DView

    # --- Internal State ---
    _last_camera_id: int
    _last_image_id: int
    _last_point3D_id: int

    def __init__(self, reconstruction_path: str, verify_integrity: bool = True, only_3d_features: bool = True) -> None:
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
            raise FileNotFoundError(f"Could not find COLMAP model in standard locations within '{reconstruction_path}'")

        self.path = model_dir

        try:
            cam_data, img_data, p3d_data = read_model(model_dir, only_3d_features=only_3d_features)
        except Exception as e:
             raise ValueError(f"Failed to load COLMAP model from '{model_dir}': {e}")

        # --- Initialize Internal Data ---
        self._num_cameras = cam_data.num_cameras
        self._camera_ids = cam_data.ids
        self._camera_model_ids = cam_data.model_ids
        self._camera_widths = cam_data.widths
        self._camera_heights = cam_data.heights
        self._camera_params = cam_data.params
        self._camera_id_to_row = {id_: i for i, id_ in enumerate(self._camera_ids)}

        self._num_images = img_data.num_images
        self._image_ids = img_data.ids
        self._image_qvecs = img_data.qvecs
        self._image_tvecs = img_data.tvecs
        self._image_camera_ids = img_data.camera_ids
        self._image_names = img_data.names # Keep as list
        self._image_id_to_row = {id_: i for i, id_ in enumerate(self._image_ids)}
        self._image_name_to_id = {name: id_ for name, id_ in zip(self._image_names, self._image_ids)}

        self._all_xys = img_data.all_xys
        self._all_point3D_ids = img_data.all_point3D_ids
        self._image_feature_indices = img_data.feature_indices

        self._num_points3D = p3d_data.num_points
        self._point_ids = p3d_data.ids
        self._point_xyzs = p3d_data.xyzs
        self._point_rgbs = p3d_data.rgbs
        self._point_errors = p3d_data.errors
        self._point3D_id_to_row = {id_: i for i, id_ in enumerate(self._point_ids)}

        self._all_track_image_ids = p3d_data.all_track_image_ids
        self._all_track_point2D_idxs = p3d_data.all_track_point2D_idxs
        self._point_track_indices = p3d_data.track_indices

        # --- Initialize API Views ---
        self.cameras = CameraView(self)
        self.images = ImageView(self)
        self.points3D = Point3DView(self)

        # --- Initialize ID counters ---
        # Use max() safely on potentially empty arrays
        self._last_camera_id = np.max(self._camera_ids) if self._num_cameras > 0 else 0
        self._last_image_id = np.max(self._image_ids) if self._num_images > 0 else 0
        self._last_point3D_id = np.max(self._point_ids) if self._num_points3D > 0 else 0

        if verify_integrity:
            errors = self._verify_consistency()
            if errors:
                print("Warning: Inconsistencies found in the loaded reconstruction:")
                for error in errors[:10]: print(f"  - {error}") # Limit output
                if len(errors) > 10: print(f"  ... ({len(errors) - 10} more)")


    def get_internal_data(self) -> InternalReconstructionData:
         """Returns the internal NumPy-based data structures."""
         # Construct the data objects on the fly from internal arrays
         cam_data = CameraData()
         cam_data.num_cameras = self._num_cameras
         cam_data.ids = self._camera_ids
         cam_data.model_ids = self._camera_model_ids
         cam_data.widths = self._camera_widths
         cam_data.heights = self._camera_heights
         cam_data.params = self._camera_params

         img_data = ImageData()
         img_data.num_images = self._num_images
         img_data.ids = self._image_ids
         img_data.qvecs = self._image_qvecs
         img_data.tvecs = self._image_tvecs
         img_data.camera_ids = self._image_camera_ids
         img_data.names = self._image_names
         img_data.all_xys = self._all_xys
         img_data.all_point3D_ids = self._all_point3D_ids
         img_data.feature_indices = self._image_feature_indices

         p3d_data = Point3DData()
         p3d_data.num_points = self._num_points3D
         p3d_data.ids = self._point_ids
         p3d_data.xyzs = self._point_xyzs
         p3d_data.rgbs = self._point_rgbs
         p3d_data.errors = self._point_errors
         p3d_data.all_track_image_ids = self._all_track_image_ids
         p3d_data.all_track_point2D_idxs = self._all_track_point2D_idxs
         p3d_data.track_indices = self._point_track_indices

         return cam_data, img_data, p3d_data

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
            internal_data = self.get_internal_data()
            write_model(internal_data, save_dir, binary)

            if output_path is not None:
                 self.path = save_dir

        except Exception as e:
             raise RuntimeError(f"Failed to save reconstruction to '{save_dir}': {e}")

    def get_image(self, image_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Image]:
        """
        Retrieves an Image object by its filename or by id.
        """
        if image_id is None and name is None:
            raise ValueError("Must provide either image_id or name.")

        image_id = image_id if image_id is not None else self._image_name_to_id.get(name)  # type: ignore[arg-type]
        return self.images.get(image_id) if image_id is not None else None

    def get_3d_points(self, points_ids: Optional[List[int]] = None, image_id: Optional[int] = None) -> List[Point3D]:
        """
        Retrieves a list of Point3D objects by their IDs or by the image ID
        """
        if points_ids is None and image_id is None:
            raise ValueError("Must provide either points_ids or image_id.")
        if points_ids is not None and image_id is not None:
            raise ValueError("Cannot provide both points_ids and image_id.")

        points_ids: List[int] = points_ids if points_ids is not None else []

        if image_id is not None:
            start, end = self._image_feature_indices[self._image_id_to_row[image_id]]
            # TODO: change class attributes to DATA
            points_ids = self._all_point3D_ids[start:end] # type: ignore[arg-type]

        return [self.points3D.get(point_id) for point_id in points_ids if point_id != INVALID_POINT3D_ID]

    # NOTE: Adding/deleting elements from NumPy arrays requires creating new arrays,
    # which can be inefficient for frequent modifications. Pre-allocation or
    # more complex data structures could improve this if needed.
    # This implementation prioritizes correctness and memory over modification speed.

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

        if model not in CAMERA_MODEL_NAMES:
            raise ValueError(f"Unknown camera model name: {model}")

        model_id = CAMERA_MODEL_NAMES[model].model_id
        num_params = CAMERA_MODEL_NAMES[model].num_params

        if len(params) != num_params:
            raise ValueError(f"Model '{model}' expects {num_params} params, got {len(params)}")

        self._last_camera_id += 1
        new_id = self._last_camera_id

        if new_id in self._camera_id_to_row:
             raise ValueError(f"Camera ID {new_id} collision detected.")

        new_row_idx = self._num_cameras

        # Append data to internal arrays (inefficiently via list conversion/concatenation)
        self._camera_ids = np.append(self._camera_ids, np.int16(new_id))
        self._camera_model_ids = np.append(self._camera_model_ids, np.int8(model_id))
        self._camera_widths = np.append(self._camera_widths, np.uint32(width))
        self._camera_heights = np.append(self._camera_heights, np.uint32(height))

        padded_params = np.zeros(MAX_CAMERA_PARAMS, dtype=np.float64)
        padded_params[:num_params] = params

        self._camera_params = np.vstack([self._camera_params, padded_params]) if self._num_cameras > 0 else padded_params.reshape(1, -1)
        self._camera_id_to_row[new_id] = new_row_idx
        self._num_cameras += 1

        return new_id


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
            raise ValueError(f"Image name '{name}' already exists.")
        if camera_id not in self._camera_id_to_row:
             raise ValueError(f"Camera ID {camera_id} does not exist.")

        self._last_image_id += 1
        new_id = self._last_image_id
        if new_id in self._image_id_to_row:
             raise ValueError(f"Image ID {new_id} collision detected.")

        new_row_idx = self._num_images
        self._image_ids = np.append(self._image_ids, np.uint16(new_id))
        self._image_qvecs = np.vstack([self._image_qvecs, np.array(qvec, dtype=np.float64)]) if self._num_images > 0 else np.array([qvec], dtype=np.float64)
        self._image_tvecs = np.vstack([self._image_tvecs, np.array(tvec, dtype=np.float64)]) if self._num_images > 0 else np.array([tvec], dtype=np.float64)
        self._image_camera_ids = np.append(self._image_camera_ids, np.uint16(camera_id))
        self._image_names.append(name) # Append to list

        # Add entry for features (initially empty)
        feature_start_idx = self._all_xys.shape[0] # Current end of concatenated features
        new_feature_indices = np.array([[feature_start_idx, feature_start_idx]], dtype=np.uint32)
        self._image_feature_indices = np.vstack([self._image_feature_indices, new_feature_indices]) if self._num_images > 0 else new_feature_indices

        self._image_id_to_row[new_id] = new_row_idx
        self._image_name_to_id[name] = new_id
        self._num_images += 1

        return new_id

    # add_point3D, delete_*, filter_* methods become significantly more complex
    # due to managing the concatenated arrays and indices. Implementing them fully
    # while maintaining efficiency requires careful handling of array manipulations
    # (insertions, deletions, index updates). This often involves creating new
    # arrays and copying data, which can negate some memory benefits if done frequently.
    # A full implementation is beyond a simple refactor here.
    # Providing stubs or simplified versions:

    # TODO: check this
    def add_point3D(self, xyz: Tuple[float, float, float],
                    track: List[Tuple[int, int]],
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
        print("Warning: add_point3D is complex with NumPy backend and not fully implemented for efficiency.")

        self._last_point3D_id += 1
        new_id = self._last_point3D_id
        if new_id in self._point3D_id_to_row:
            raise ValueError(f"Point3D ID {new_id} collision.")

        # **Simplified approach (Potentially Inefficient):**
        # 1. Append basic point data (xyz, rgb, error)
        # 2. Append track data to concatenated track arrays
        # 3. Calculate new track index entry
        # 4. Update the _all_point3D_ids in the relevant image features (VERY hard with current structure)

        # --- Basic Data Append ---
        new_row_idx = self._num_points3D
        self._point_ids = np.append(self._point_ids, np.int64(new_id))
        self._point_xyzs = np.vstack([self._point_xyzs, np.array(xyz, dtype=np.float64)]) if self._num_points3D > 0 else np.array([xyz], dtype=np.float64)
        self._point_rgbs = np.vstack([self._point_rgbs, np.array(rgb, dtype=np.uint8)]) if self._num_points3D > 0 else np.array([rgb], dtype=np.uint8)
        self._point_errors = np.append(self._point_errors, np.float64(error))

        # --- Track Data Append ---
        track_start_idx = self._all_track_image_ids.shape[0]
        track_img_ids = [t[0] for t in track]
        track_p2d_idxs = [t[1] for t in track]
        self._all_track_image_ids = np.concatenate([self._all_track_image_ids, np.array(track_img_ids, dtype=np.int64)])
        self._all_track_point2D_idxs = np.concatenate([self._all_track_point2D_idxs, np.array(track_p2d_idxs, dtype=np.int64)])
        track_end_idx = self._all_track_image_ids.shape[0]
        new_track_indices = np.array([[track_start_idx, track_end_idx]], dtype=np.int64)
        self._point_track_indices = np.vstack([self._point_track_indices, new_track_indices]) if self._num_points3D > 0 else new_track_indices

        # --- Update ID Mapping ---
        self._point3D_id_to_row[new_id] = new_row_idx
        self._num_points3D += 1

        # --- CRITICAL MISSING PIECE: Update Image Features ---
        # Need to find the correct indices in _all_point3D_ids corresponding
        # to the (image_id, p2d_idx) pairs in the track and set them to new_id.
        # This requires searching or more complex indexing, making it slow.
        print(f"Warning: Image feature back-references for new Point3D {new_id} not updated.")
        # Example placeholder logic (inefficient):
        # for img_id, p2d_idx in track:
        #     if img_id in self.images:
        #         img_row = self._image_id_to_row[img_id]
        #         feat_start, feat_end = self._image_feature_indices[img_row]
        #         # This requires knowing the *original* p2d_idx within the *original* full feature list
        #         # The p2d_idx in the track refers to this *original* index.
        #         # Finding the corresponding index in the *concatenated* array is hard.
        #         pass # Complex logic needed here

        return new_id


    def delete_camera(self, camera_id: int, force: bool = False) -> None:
        """Deletes a camera. Requires complex array manipulation."""
        print("Warning: delete_camera not fully implemented due to complexity.")
        if camera_id not in self._camera_id_to_row: return

        # Check usage by images
        images_using_camera = self._image_ids[self._image_camera_ids == camera_id]
        if len(images_using_camera) > 0 and not force:
            raise ValueError(f"Camera {camera_id} used by images (e.g., {images_using_camera[0]}). Use force=True.")

        # If force=True, need to delete associated images first (recursive call)
        # Then, remove camera row from arrays and update _camera_id_to_row
        # This requires creating masks, slicing, and rebuilding the map.


    def delete_image(self, image_id: int) -> None:
        """Deletes an image. Requires complex array manipulation."""
        print("Warning: delete_image not fully implemented due to complexity.")
        if image_id not in self._image_id_to_row: return

        # Needs to:
        # 1. Remove image row from _image_* arrays.
        # 2. Remove corresponding features from _all_xys, _all_point3D_ids.
        # 3. Update _image_feature_indices for subsequent images.
        # 4. Remove observations from _all_track_* arrays for points observed by this image.
        # 5. Update _point_track_indices for affected points.
        # 6. Potentially delete points whose tracks become empty.
        # 7. Update _image_id_to_row, _image_name_to_id.

    def delete_point3D(self, point3D_id: int) -> None:
        """Deletes a 3D point. Requires complex array manipulation."""
        print("Warning: delete_point3D not fully implemented due to complexity.")
        if point3D_id not in self._point3D_id_to_row: return

        # Needs to:
        # 1. Remove point row from _point_* arrays.
        # 2. Remove track from _all_track_* arrays.
        # 3. Update _point_track_indices for subsequent points.
        # 4. Invalidate references in _all_point3D_ids for features that pointed here. (Hard part)
        # 5. Update _point3D_id_to_row.


    def filter_points3D(self,
                       min_track_len: int = 2,
                       max_error: float = float('inf'),
                       min_angle: float = 0.0) -> int:
        """
        Filters the 3D point cloud based on criteria. Operates on NumPy arrays.
        NOTE: Deletion itself is complex; this implementation identifies points
              to delete but doesn't perform the complex array updates.
        """
        if self._num_points3D == 0: return 0

        points_to_delete_mask = np.zeros(self._num_points3D, dtype=bool)

        # Criterion 1 & 2: Track Length and Error (Vectorized)
        track_lengths = self._point_track_indices[:, 1] - self._point_track_indices[:, 0]
        points_to_delete_mask |= (track_lengths < min_track_len)
        points_to_delete_mask |= (self._point_errors > max_error)

        # Criterion 3: Triangulation Angle (Requires iteration)
        if min_angle > 0.0:
             # Pre-calculate all camera centers (avoids repeated calculation)
             cam_centers = {}
             for img_id in self.images.keys(): # Iterate through available image IDs
                try:
                    # Get on-demand image object to calculate center
                    img = self.images[img_id]
                    cam_centers[img_id] = img.get_camera_center()
                except KeyError: # Should not happen if keys() is correct
                    continue

             for i in range(self._num_points3D):
                 if points_to_delete_mask[i]: continue # Already marked for deletion

                 track_start, track_end = self._point_track_indices[i]
                 if track_end - track_start < 2: continue # Need >= 2 obs for angle

                 point_xyz = self._point_xyzs[i]
                 observing_img_ids = self._all_track_image_ids[track_start:track_end]

                 valid_centers = [cam_centers[img_id] for img_id in observing_img_ids if img_id in cam_centers]

                 if len(valid_centers) < 2: continue

                 max_observed_angle = 0.0
                 for j1 in range(len(valid_centers)):
                     for j2 in range(j1 + 1, len(valid_centers)):
                         angle = angle_between_rays(valid_centers[j1], valid_centers[j2], point_xyz)
                         max_observed_angle = max(max_observed_angle, angle)

                 if max_observed_angle < min_angle:
                     points_to_delete_mask[i] = True


        # --- Actual Deletion ---
        # This is the complex part. A robust implementation would involve:
        # 1. Getting the IDs of points where points_to_delete_mask is True.
        # 2. Calling delete_point3D for each ID (which needs full implementation).
        # Alternatively, create new arrays containing only the points *not* masked,
        # and carefully rebuild all index arrays and concatenated data.
        num_to_delete = np.sum(points_to_delete_mask)
        if num_to_delete > 0:
             print(f"Warning: Identified {num_to_delete} points to filter, but deletion is not fully implemented.")
             # Placeholder: Just update the count for now, data remains.
             # A real implementation would modify the arrays here.
             # self._num_points3D -= num_to_delete
             # Update _point_ids, _point_xyzs, etc. using boolean indexing:
             # self._point_ids = self._point_ids[~points_to_delete_mask]
             # ... and so on for all point arrays ...
             # Rebuilding concatenated arrays and indices is the hard part.

        return num_to_delete # Return number identified

    def get_statistics(self) -> Dict[str, float]:
         """Calculates basic statistics about the reconstruction."""
         num_cameras = float(self._num_cameras)
         num_images = float(self._num_images)
         num_points = float(self._num_points3D)

         mean_track_length = 0.0
         mean_reprojection_error = 0.0
         if num_points > 0:
             track_lengths = self._point_track_indices[:, 1] - self._point_track_indices[:, 0]
             mean_track_length = np.mean(track_lengths) if len(track_lengths) > 0 else 0.0
             mean_reprojection_error = np.mean(self._point_errors) if len(self._point_errors) > 0 else 0.0

         mean_observations = 0.0
         mean_valid_observations = 0.0
         if num_images > 0:
             feature_counts = self._image_feature_indices[:, 1] - self._image_feature_indices[:, 0]
             mean_observations = np.mean(feature_counts) if len(feature_counts) > 0 else 0.0
             # Calculating valid observations requires iterating through _all_point3D_ids segments
             total_valid_obs = np.sum(self._all_point3D_ids != INVALID_POINT3D_ID)
             mean_valid_observations = total_valid_obs / num_images

         return {
             "num_cameras": num_cameras,
             "num_images": num_images,
             "num_points3D": num_points,
             "mean_track_length": mean_track_length,
             "mean_observations_per_image": mean_observations,
             "mean_valid_observations_per_image": mean_valid_observations,
             "mean_reprojection_error": mean_reprojection_error,
         }


    def __str__(self) -> str:
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

    def _verify_consistency(self, max_workers: Optional[int] = None) -> List[str]:
        """Performs internal consistency checks on the NumPy data structures."""
        # This check needs to be completely rewritten for the NumPy backend.
        # It involves checking index bounds, ID consistency, back-references etc.
        # Example checks:
        errors = []
        if self._num_cameras > 0:
             if not (len(self._camera_ids) == self._num_cameras and
                     len(self._camera_model_ids) == self._num_cameras and
                     len(self._camera_widths) == self._num_cameras and
                     len(self._camera_heights) == self._num_cameras and
                     self._camera_params.shape[0] == self._num_cameras):
                 errors.append("Camera array length mismatch.")
             if len(self._camera_id_to_row) != self._num_cameras:
                 errors.append("Camera ID map size mismatch.")
             # Check model IDs are valid
             if not np.all(np.isin(self._camera_model_ids, list(CAMERA_MODEL_IDS.keys()))):
                 errors.append("Invalid camera model ID detected.")

        if self._num_images > 0:
             # Check lengths of image arrays
             if not (len(self._image_ids) == self._num_images and
                      self._image_qvecs.shape[0] == self._num_images and
                      self._image_tvecs.shape[0] == self._num_images and
                      len(self._image_camera_ids) == self._num_images and
                      len(self._image_names) == self._num_images and
                      self._image_feature_indices.shape[0] == self._num_images):
                 errors.append("Image array length mismatch.")
             if len(self._image_id_to_row) != self._num_images:
                 errors.append("Image ID map size mismatch.")
             if len(self._image_name_to_id) != self._num_images:
                 errors.append("Image name map size mismatch.")
             # Check camera IDs exist
             if not np.all(np.isin(self._image_camera_ids, self._camera_ids)):
                  errors.append("Image references non-existent camera ID.")
             # Check feature indices are valid and monotonically increasing
             if np.any(self._image_feature_indices[:, 0] > self._image_feature_indices[:, 1]):
                  errors.append("Invalid image feature index range (start > end).")
             if np.any(self._image_feature_indices[:-1, 1] > self._image_feature_indices[1:, 0]):
                  # This check assumes images are sorted by feature index, which might not be true if modified
                  # A better check is that ranges don't overlap and ends don't exceed array bounds
                  pass
             if self._image_feature_indices.size > 0 and self._image_feature_indices[-1, 1] > self._all_xys.shape[0]:
                 errors.append("Image feature indices exceed bounds of feature data arrays.")

        # TODO: Add similar checks for Point3D data, track indices, and crucially,
        #       the consistency between _all_point3D_ids and the point tracks.
        #       This back-reference check is the most complex.

        print(f"Warning: _verify_consistency is partially implemented for NumPy backend.")
        return errors

