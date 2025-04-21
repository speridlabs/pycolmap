import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional 

from .image import Image
from .camera import Camera
from .point3d import Point3D

from .recon_utils import CameraView, ImageView, Point3DView

from .io.binary import MAX_CAMERA_PARAMS
from .io import read_model, write_model, CameraData, ImageData, Point3DData, InternalReconstructionData

from .utils import find_model_path, angle_between_rays, qvec2rotmat, rotmat2qvec
from .types import INVALID_POINT3D_ID, CAMERA_MODEL_NAMES

# NOTE: some other data structures may be more suitable for this kind of operations
# NOTE: we can use numba to speed up some of the operations
# NOTE: need to refactor to use xData classes directly
# TODO: implement more utility functions to manipulate the data

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
    _camera_ids: NDArray[np.uint16]
    _camera_model_ids: NDArray[np.uint8]
    _camera_widths: NDArray[np.uint32]
    _camera_heights: NDArray[np.uint32]
    _camera_params: NDArray[np.float64] # Padded (N, MAX_CAMERA_PARAMS)
    _camera_id_to_row: Dict[int, int]
    _num_cameras: int

    # Images
    _image_ids: NDArray[np.uint16]
    _image_qvecs: NDArray[np.float64] # (N, 4)
    _image_tvecs: NDArray[np.float64] # (N, 3)
    _image_camera_ids: NDArray[np.uint16]
    _image_names: List[str]
    _image_id_to_row: Dict[int, int]
    _image_name_to_id: Dict[str, int]
    _num_images: int

    # Image Features (Concatenated)
    _all_xys: NDArray[np.float64] # (TotalPoints, 2)
    _all_point3D_ids: NDArray[np.uint32] # (TotalPoints,) - Note: COLMAP uses uint64 internally, but data class uses uint32
    _image_feature_indices: NDArray[np.uint32] # (N_images, 2) [start, end)

    # 3D Points
    _point_ids: NDArray[np.uint32] # Note: COLMAP uses uint64 internally, but data class uses uint32
    _point_xyzs: NDArray[np.float64] # (M, 3)
    _point_rgbs: NDArray[np.uint8] # (M, 3)
    _point_errors: NDArray[np.float64] # (M,)
    _point3D_id_to_row: Dict[int, int]
    _num_points3D: int

    # Point Tracks (Concatenated)
    _all_track_image_ids: NDArray[np.uint16] # (TotalTrackLen,)
    _all_track_point2D_idxs: NDArray[np.uint32] # (TotalTrackLen,)
    _point_track_indices: NDArray[np.uint32] # (M, 2) [start, end)

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
        self._last_camera_id = int(np.max(self._camera_ids) if self._num_cameras > 0 else 0)
        self._last_image_id = int(np.max(self._image_ids) if self._num_images > 0 else 0)
        self._last_point3D_id = int(np.max(self._point_ids) if self._num_points3D > 0 else 0)

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

        _points_ids: List[int] = points_ids if points_ids is not None else []

        if image_id is not None:
            start, end = self._image_feature_indices[self._image_id_to_row[image_id]]
            _points_ids = self._all_point3D_ids[start:end] # type: ignore[arg-type]

        return [self.points3D.get(point_id) for point_id in _points_ids if point_id != INVALID_POINT3D_ID]

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
                  tvec: Tuple[float, float, float] = (0.0, 0.0, 0.0), 
                  image_id: Optional[int] = None) -> int:
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

        if not image_id:
            self._last_image_id += 1
            image_id= self._last_image_id
        else:
            self._last_image_id = max(self._last_image_id, image_id)

        if image_id in self._image_id_to_row:
             raise ValueError(f"Image ID {image_id} collision detected.")

        new_row_idx = self._num_images
        self._image_ids = np.append(self._image_ids, np.uint16(image_id))
        self._image_qvecs = np.vstack([self._image_qvecs, np.array(qvec, dtype=np.float64)]) if self._num_images > 0 else np.array([qvec], dtype=np.float64)
        self._image_tvecs = np.vstack([self._image_tvecs, np.array(tvec, dtype=np.float64)]) if self._num_images > 0 else np.array([tvec], dtype=np.float64)
        self._image_camera_ids = np.append(self._image_camera_ids, np.uint16(camera_id))
        self._image_names.append(name) # Append to list

        # Add entry for features (initially empty)
        feature_start_idx = self._all_xys.shape[0] # Current end of concatenated features
        new_feature_indices = np.array([[feature_start_idx, feature_start_idx]], dtype=np.uint32)
        self._image_feature_indices = np.vstack([self._image_feature_indices, new_feature_indices]) if self._num_images > 0 else new_feature_indices

        self._image_id_to_row[image_id] = new_row_idx
        self._image_name_to_id[name] = image_id 
        self._num_images += 1

        return image_id

    def add_3d_points(self, points_data: List[Tuple[Tuple[float, float, float], List[Tuple[int, int]], Tuple[int, int, int], float]]) -> List[int]:
        """
        Adds multiple new 3D points to the reconstruction efficiently.

        Assigns new unique point IDs. Updates the `_all_point3D_ids` array
        in the corresponding Image features based on the tracks.

        Args:
            points_data: A list where each element is a tuple containing data
                         for one point: `(xyz, track, rgb, error)`.
                         - xyz: Tuple[float, float, float]
                         - track: List[Tuple[int, int]] (image_id, point2D_idx)
                         - rgb: Tuple[int, int, int]
                         - error: float

        Returns:
            A list of the newly assigned 3D point IDs.

        Raises:
            ValueError: If validation fails for any point or its track:
                        - An image_id in a track doesn't exist.
                        - A point2D_idx is out of bounds for its image.
                        - A 2D feature (image_id, point2D_idx) is already
                          assigned to an existing 3D point.
                        - A 2D feature is assigned to multiple points within
                          this batch add operation.
        """
        if not points_data: return []

        num_new_points = len(points_data)
        new_ids = np.arange(self._last_point3D_id + 1, self._last_point3D_id + 1 + num_new_points, dtype=np.uint32)
        self._last_point3D_id += num_new_points # Reserve IDs immediately

        # lists to collect validated data
        new_xyzs_list = []
        new_rgbs_list = []
        new_errors_list = []
        new_track_indices_list = [] # Stores (start, end) for each new point's track

        # lists to collect concatenated track data
        new_track_img_ids_list = []
        new_track_p2d_idxs_list = []

        # Track features to update
        # Stores {global_feature_index: new_point3D_id}
        features_to_update: Dict[int, int] = {}
        current_track_offset = len(self._all_track_image_ids) # Start index for new tracks in the concatenated array

        # --- Validation Loop ---
        for i, point_input in enumerate(points_data):
            xyz, track, rgb, error = point_input
            new_point_id = new_ids[i]
            point_track_img_ids = []
            point_track_p2d_idxs = []

            if not isinstance(track, list):
                 raise ValueError(f"Point {i}: Track must be a list of (image_id, point2D_idx) tuples.")

            # Validate track elements for the current point
            for img_id, p2d_idx in track:
                if img_id not in self._image_id_to_row:
                    # Rollback reserved IDs if validation fails mid-batch
                    self._last_point3D_id -= num_new_points
                    raise ValueError(f"Point {i} (Prospective ID {new_point_id}): Image ID {img_id} in track does not exist.")

                img_row = self._image_id_to_row[img_id]
                feat_start, feat_end = self._image_feature_indices[img_row]
                num_features_in_image = feat_end - feat_start

                if not (0 <= p2d_idx < num_features_in_image):
                    self._last_point3D_id -= num_new_points
                    raise ValueError(
                        f"Point {i} (Prospective ID {new_point_id}): Point2D index {p2d_idx} is out of bounds "
                        f"for image {img_id} (size: {num_features_in_image})."
                    )

                global_feat_idx = feat_start + p2d_idx

                # Check if feature is already assigned (either existing or in this batch)
                # Accessing _all_point3D_ids is safe as we haven't modified it yet
                existing_p3d_id = self._all_point3D_ids[global_feat_idx]
                if existing_p3d_id != INVALID_POINT3D_ID:
                     self._last_point3D_id -= num_new_points
                     raise ValueError(
                        f"Point {i} (Prospective ID {new_point_id}): 2D feature at index {p2d_idx} in image {img_id} "
                        f"is already assigned to Point3D ID {existing_p3d_id}."
                    )
                if global_feat_idx in features_to_update:
                     self._last_point3D_id -= num_new_points
                     raise ValueError(
                        f"Point {i} (Prospective ID {new_point_id}): 2D feature at index {p2d_idx} in image {img_id} "
                        f"is assigned to multiple new points in this batch (new IDs {features_to_update[global_feat_idx]} and {new_point_id})."
                    )

                # Mark feature for update and collect track info for this point
                features_to_update[global_feat_idx] = new_point_id
                point_track_img_ids.append(img_id)
                point_track_p2d_idxs.append(p2d_idx)

            # If validation passed for this point, collect its data
            new_xyzs_list.append(xyz)
            new_rgbs_list.append(rgb)
            new_errors_list.append(error)

            # Add this point's track data to the main lists for concatenation
            track_len = len(point_track_img_ids)
            new_track_img_ids_list.extend(point_track_img_ids)
            new_track_p2d_idxs_list.extend(point_track_p2d_idxs)

            # Calculate and store track indices for this point relative to the concatenated array
            track_start_idx = current_track_offset
            track_end_idx = current_track_offset + track_len
            new_track_indices_list.append([track_start_idx, track_end_idx])
            current_track_offset = track_end_idx # Update offset for the next point's track start

        # Convert collected data to NumPy arrays
        # Ensure correct dtypes, especially for potentially empty initial arrays
        new_xyzs = np.array(new_xyzs_list, dtype=np.float64)
        new_rgbs = np.array(new_rgbs_list, dtype=np.uint8)
        new_errors = np.array(new_errors_list, dtype=np.float64)
        new_track_indices = np.array(new_track_indices_list, dtype=np.uint32)
        new_track_img_ids = np.array(new_track_img_ids_list, dtype=np.uint16)
        new_track_p2d_idxs = np.array(new_track_p2d_idxs_list, dtype=np.uint32)

        # Append data to internal arrays
        # Use concatenate for 1D arrays and vstack for 2D arrays
        self._point_ids = np.concatenate((self._point_ids, new_ids))
        self._point_xyzs = np.vstack((self._point_xyzs, new_xyzs)) if self._num_points3D > 0 else new_xyzs
        self._point_rgbs = np.vstack((self._point_rgbs, new_rgbs)) if self._num_points3D > 0 else new_rgbs
        self._point_errors = np.concatenate((self._point_errors, new_errors))
        self._point_track_indices = np.vstack((self._point_track_indices, new_track_indices)) if self._num_points3D > 0 else new_track_indices

        # Check if track arrays were initially empty
        if len(self._all_track_image_ids) > 0:
            self._all_track_image_ids = np.concatenate((self._all_track_image_ids, new_track_img_ids))
            self._all_track_point2D_idxs = np.concatenate((self._all_track_point2D_idxs, new_track_p2d_idxs))
        else:
            self._all_track_image_ids = new_track_img_ids
            self._all_track_point2D_idxs = new_track_p2d_idxs

        # Update Image Features
        # Apply the collected updates to _all_point3D_ids
        update_indices = np.array(list(features_to_update.keys()), dtype=np.int64)
        update_values = np.array(list(features_to_update.values()), dtype=np.int64)

        if len(update_indices) > 0:
            # Ensure indices are within bounds (should be, due to validation)
            if np.max(update_indices) >= len(self._all_point3D_ids):
                 # This indicates an internal logic error if it happens
                 self._last_point3D_id -= num_new_points # Attempt rollback
                 raise IndexError("Internal error: Feature index for update is out of bounds after validation.")
            self._all_point3D_ids[update_indices] = update_values

        # Update Count and Mapping
        self._num_points3D += num_new_points
        # Rebuild mapping is crucial
        self._point3D_id_to_row = {id_: i for i, id_ in enumerate(self._point_ids)}

        del features_to_update # Cleanup

        return new_ids.tolist()


    def delete_camera(self, camera_id: int, force: bool = False) -> None:
        """
        Deletes a camera from the reconstruction.

        Args:
            camera_id: The ID of the camera to delete.
            force: If True, also deletes all images using this camera. If False
                   (default), raises an error if the camera is still in use.

        Returns: None

        Raises:
            ValueError: If `force` is False and the camera is still used by images.
        """
        if camera_id not in self._camera_id_to_row: return

        # Check camera usage by images
        images_using_camera = self._image_ids[self._image_camera_ids == camera_id]

        if len(images_using_camera) > 0:
            if not force:
                raise ValueError(
                    f"Camera {camera_id} is used by {len(images_using_camera)} images "
                    f"(e.g., ID {images_using_camera[0]}). Use force=True to delete images too."
                )

            for img_id in images_using_camera:
                self.delete_image(img_id)

        # Proceed with deletion
        # Have in mind that if the ele
        row = self._camera_id_to_row[camera_id]
        self._camera_ids = np.delete(self._camera_ids, row)
        self._camera_model_ids = np.delete(self._camera_model_ids, row)
        self._camera_widths = np.delete(self._camera_widths, row)
        self._camera_heights = np.delete(self._camera_heights, row)
        self._camera_params = np.delete(self._camera_params, row, axis=0)

        self._num_cameras -= 1
        # Rebuild the ID-to-row mapping (could be slow for large numbers of cameras)
        # A more optimized approach might update indices directly, but this is simpler
        self._camera_id_to_row = {id_: i for i, id_ in enumerate(self._camera_ids)}


    def delete_image(self, image_id: Optional[int]=None, image_name: Optional[str]=None) -> None:
        """
        Deletes an image from the reconstruction.

        Also removes this image's observations from the tracks of all 3D points
        it observes. If removing an observation causes a 3D point's track to
        become empty, that 3D point is also deleted.

        Args:
            image_id: (Optional) The ID of the image to delete.
            image_name: (Optional) The name of the image to delete. If both `image_id` and

        Returns: None
        """
        if image_id is None and image_name is None:
            raise ValueError("Must provide either image_id or image_name.")
        if image_name is not None and image_id is not None:
            raise ValueError("Cannot provide both image_id and image_name.")
        if image_name is not None: 
            image_id = self._image_name_to_id.get(image_name)
        if image_id is None or image_id not in self._image_id_to_row: return

        row_idx = self._image_id_to_row[image_id]
        image_name = self._image_names[row_idx] # Get name before list modification

        # Identify Features
        feat_start, feat_end = self._image_feature_indices[row_idx]
        num_features = feat_end - feat_start
        feature_indices_to_delete = np.arange(feat_start, feat_end)

        # Find Track Entries to Delete 
        # Find indices in the global track arrays corresponding to this image
        track_deletion_mask = (self._all_track_image_ids == image_id)
        track_indices_to_delete = np.where(track_deletion_mask)[0]
        num_tracks_deleted = len(track_indices_to_delete)

        if num_tracks_deleted > 0:
            # Calculate Index Shifts for Point Tracks 
            # How many deleted track entries were before the start/end of each point's track?
            # Use searchsorted for efficiency
            original_track_starts = self._point_track_indices[:, 0]
            original_track_ends = self._point_track_indices[:, 1]
            deleted_counts_before_start = np.searchsorted(track_indices_to_delete, original_track_starts)
            deleted_counts_before_end = np.searchsorted(track_indices_to_delete, original_track_ends)

            # Delete Tracks 
            self._all_track_image_ids = np.delete(self._all_track_image_ids, track_indices_to_delete)
            self._all_track_point2D_idxs = np.delete(self._all_track_point2D_idxs, track_indices_to_delete)

            # Update Point Track Indices
            self._point_track_indices[:, 0] -= deleted_counts_before_start
            self._point_track_indices[:, 1] -= deleted_counts_before_end

        # Identify Points to Delete (Empty Tracks)
        empty_track_mask = (self._point_track_indices[:, 1] == self._point_track_indices[:, 0])
        point_rows_to_delete = np.where(empty_track_mask)[0]
        point_ids_to_delete = self._point_ids[point_rows_to_delete]

        if len(point_ids_to_delete) > 0:
            # Delete Empty Points 
            self._point_ids = np.delete(self._point_ids, point_rows_to_delete)
            self._point_xyzs = np.delete(self._point_xyzs, point_rows_to_delete, axis=0)
            self._point_rgbs = np.delete(self._point_rgbs, point_rows_to_delete, axis=0)
            self._point_errors = np.delete(self._point_errors, point_rows_to_delete)
            self._point_track_indices = np.delete(self._point_track_indices, point_rows_to_delete, axis=0) # Delete their track index entries
            self._num_points3D -= len(point_rows_to_delete)

            # Rebuild mapping is crucial after deleting points
            self._point3D_id_to_row = {id_: i for i, id_ in enumerate(self._point_ids)}

            # Invalidate References in Remaining Features
            # Find features (across all images) pointing to the deleted points
            invalidation_mask = np.isin(self._all_point3D_ids, point_ids_to_delete)
            self._all_point3D_ids[invalidation_mask] = INVALID_POINT3D_ID

        # Delete Image Features 
        if num_features > 0:
            self._all_xys = np.delete(self._all_xys, feature_indices_to_delete, axis=0)
            # Make sure to delete from the potentially updated _all_point3D_ids (after invalidation)
            self._all_point3D_ids = np.delete(self._all_point3D_ids, feature_indices_to_delete)

        # Update Image Feature Indices
        # Decrement indices for images *after* the deleted one
        image_rows_after = np.arange(self._num_images) > row_idx
        self._image_feature_indices[image_rows_after, :] -= num_features
        # Remove the row for the deleted image
        self._image_feature_indices = np.delete(self._image_feature_indices, row_idx, axis=0)

        # Delete Image Metadata
        self._image_ids = np.delete(self._image_ids, row_idx)
        self._image_qvecs = np.delete(self._image_qvecs, row_idx, axis=0)
        self._image_tvecs = np.delete(self._image_tvecs, row_idx, axis=0)
        self._image_camera_ids = np.delete(self._image_camera_ids, row_idx)
        del self._image_names[row_idx]

        # Update Mappings & Count
        self._num_images -= 1
        if image_name in self._image_name_to_id:
             del self._image_name_to_id[image_name]
        # Rebuild ID mapping is crucial as row indices have changed
        self._image_id_to_row = {id_: i for i, id_ in enumerate(self._image_ids)}


    def delete_3d_points(self, point3D_ids: List[int]) -> None:
        """
        Deletes multiple 3D points from the reconstruction and updates associated data.

        Removes point metadata, their tracks from the global track arrays, and
        invalidates references to these points in image features. Updates all
        internal NumPy arrays and index mappings accordingly.

        Args:
            point3D_ids: A list of 3D point IDs to delete.

        Returns: None
        """
        
        if not point3D_ids or len(point3D_ids) == 0 or  self._num_points3D == 0:
            return

        # Filter Valid ids & Get Rows
        point_ids_to_delete_set = set(point3D_ids)
        valid_point_rows_mask = np.isin(self._point_ids, list(point_ids_to_delete_set))
        point_rows_to_delete = np.where(valid_point_rows_mask)[0]

        if len(point_rows_to_delete) == 0: return # None of the requested IDs exist

        actual_point_ids_deleted = self._point_ids[point_rows_to_delete]
        track_ranges_to_delete = self._point_track_indices[point_rows_to_delete]

        # Identify Track Indices to Delete 
        starts = track_ranges_to_delete[:, 0]
        ends = track_ranges_to_delete[:, 1]
        lengths = ends - starts
        valid_mask = lengths > 0

        if not np.any(valid_mask):
            # Points existed but had no tracks, or all tracks were empty
            track_indices_to_delete_arr = np.array([], dtype=np.int64)
            all_track_indices_to_delete = False
        else:
            valid_starts = starts[valid_mask]
            valid_lengths = lengths[valid_mask]
            total_tracks_deleted = np.sum(valid_lengths)

            # Generate all indices in one go
            repeated_starts = np.repeat(valid_starts, valid_lengths)
            range_counters = np.arange(total_tracks_deleted)
            cumulative_lengths = np.cumsum(valid_lengths)
            start_of_each_range_in_counter = np.concatenate(([0], cumulative_lengths[:-1]))
            repeated_start_of_range = np.repeat(start_of_each_range_in_counter, valid_lengths)
            relative_indices = range_counters - repeated_start_of_range
            track_indices_to_delete_arr = repeated_starts + relative_indices

            # Ensure sorted order
            track_indices_to_delete_arr.sort()
            all_track_indices_to_delete = True

        # Calculate Index Shifts for Remaining Points
        if all_track_indices_to_delete:
            # Get track indices for points *not* being deleted
            point_rows_to_keep = np.where(~valid_point_rows_mask)[0]
            original_track_starts_kept = self._point_track_indices[point_rows_to_keep, 0]
            original_track_ends_kept = self._point_track_indices[point_rows_to_keep, 1]

            # Calculate how many deleted tracks were before the start/end of kept tracks
            deleted_counts_before_start = np.searchsorted(track_indices_to_delete_arr, original_track_starts_kept)
            deleted_counts_before_end = np.searchsorted(track_indices_to_delete_arr, original_track_ends_kept)

            # Delete Tracks using Boolean Masking (faster)
            # Create a mask to keep elements *not* in track_indices_to_delete_arr
            num_total_tracks = len(self._all_track_image_ids)
            track_keep_mask = np.ones(num_total_tracks, dtype=bool)
            track_keep_mask[track_indices_to_delete_arr] = False

            # Apply the mask to create new arrays
            self._all_track_image_ids = self._all_track_image_ids[track_keep_mask]
            self._all_track_point2D_idxs = self._all_track_point2D_idxs[track_keep_mask]

        # Invalidate Feature References
        # Find features pointing to *any* of the deleted points
        invalidation_mask = np.isin(self._all_point3D_ids, actual_point_ids_deleted)
        self._all_point3D_ids[invalidation_mask] = INVALID_POINT3D_ID

        # Update Point Metadata
        self._point_ids = self._point_ids[~valid_point_rows_mask]
        self._point_xyzs = self._point_xyzs[~valid_point_rows_mask]
        self._point_rgbs = self._point_rgbs[~valid_point_rows_mask]
        self._point_errors = self._point_errors[~valid_point_rows_mask]

        # Update Point Track Indices
        # Keep only rows for remaining points
        self._point_track_indices = self._point_track_indices[~valid_point_rows_mask]

        # Adjust indices if tracks were deleted
        if all_track_indices_to_delete:
            self._point_track_indices[:, 0] -= deleted_counts_before_start # type: ignore[assignment]
            self._point_track_indices[:, 1] -= deleted_counts_before_end # type: ignore[assignment]

        # Update Count and Mapping
        num_deleted = len(point_rows_to_delete)
        self._num_points3D -= num_deleted

        # Rebuild mapping is crucial
        self._point3D_id_to_row = {id_: i for i, id_ in enumerate(self._point_ids)}


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
        raise NotImplementedError("Filtering is not implemented for NumPy backend.")

    def get_statistics(self) -> Dict[str, float]:
         """Calculates basic statistics about the reconstruction."""
         num_cameras = float(self._num_cameras)
         num_images = float(self._num_images)
         num_points = float(self._num_points3D)

         mean_track_length = 0.0
         mean_reprojection_error = 0.0
         if num_points > 0:
             track_lengths = self._point_track_indices[:, 1] - self._point_track_indices[:, 0]
             mean_track_length = float(np.mean(track_lengths)) if len(track_lengths) > 0 else 0.0
             mean_reprojection_error = float(np.mean(self._point_errors)) if len(self._point_errors) > 0 else 0.0

         mean_observations = 0.0
         mean_valid_observations = 0.0
         if num_images > 0:
             feature_counts = self._image_feature_indices[:, 1] - self._image_feature_indices[:, 0]
             mean_observations = float(np.mean(feature_counts)) if len(feature_counts) > 0 else 0.0
             # Calculating valid observations requires iterating through _all_point3D_ids segments
             total_valid_obs = np.sum(self._all_point3D_ids != INVALID_POINT3D_ID)
             mean_valid_observations = float(total_valid_obs / num_images) if num_images > 0 else 0.0 # Added check for division by zero

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

    def _verify_consistency(self) -> List[str]:
        """
        Performs internal consistency checks on the NumPy arrays. Useful for debugging.

        Returns:
            List of error messages. Empty if no errors found.
        """
        errors: List[str] = []
        prefix = "Consistency check failed: "

        # --- Basic Count Checks ---
        if self._num_cameras != len(self._camera_ids):
            errors.append(f"{prefix} _num_cameras ({self._num_cameras}) != len(_camera_ids) ({len(self._camera_ids)})")
        if self._num_images != len(self._image_ids):
            errors.append(f"{prefix} _num_images ({self._num_images}) != len(_image_ids) ({len(self._image_ids)})")
        if self._num_points3D != len(self._point_ids):
            errors.append(f"{prefix} _num_points3D ({self._num_points3D}) != len(_point_ids) ({len(self._point_ids)})")

        # --- Array Length Checks ---
        cam_arrays = {
            "_camera_model_ids": self._camera_model_ids,
            "_camera_widths": self._camera_widths,
            "_camera_heights": self._camera_heights,
            "_camera_params": self._camera_params,
        }
        for name, arr in cam_arrays.items():
            if arr.shape[0] != self._num_cameras:
                errors.append(f"{prefix} len({name}) ({arr.shape[0]}) != _num_cameras ({self._num_cameras})")

        img_arrays = {
            "_image_qvecs": self._image_qvecs,
            "_image_tvecs": self._image_tvecs,
            "_image_camera_ids": self._image_camera_ids,
            "_image_feature_indices": self._image_feature_indices,
        }
        for name, arr in img_arrays.items():
             if arr.shape[0] != self._num_images:
                 errors.append(f"{prefix} len({name}) ({arr.shape[0]}) != _num_images ({self._num_images})")
        if len(self._image_names) != self._num_images:
             errors.append(f"{prefix} len(_image_names) ({len(self._image_names)}) != _num_images ({self._num_images})")

        p3d_arrays = {
            "_point_xyzs": self._point_xyzs,
            "_point_rgbs": self._point_rgbs,
            "_point_errors": self._point_errors,
            "_point_track_indices": self._point_track_indices,
        }
        for name, arr in p3d_arrays.items():
             if arr.shape[0] != self._num_points3D:
                 errors.append(f"{prefix} len({name}) ({arr.shape[0]}) != _num_points3D ({self._num_points3D})")

        # --- ID Mapping Checks ---
        if set(self._camera_id_to_row.keys()) != set(self._camera_ids):
            errors.append(f"{prefix} _camera_id_to_row keys do not match _camera_ids")
        if set(self._image_id_to_row.keys()) != set(self._image_ids):
            errors.append(f"{prefix} _image_id_to_row keys do not match _image_ids")
        if set(self._point3D_id_to_row.keys()) != set(self._point_ids):
            errors.append(f"{prefix} _point3D_id_to_row keys do not match _point_ids")
        if set(self._image_name_to_id.keys()) != set(self._image_names):
             errors.append(f"{prefix} _image_name_to_id keys do not match _image_names")
        if set(self._image_name_to_id.values()) != set(self._image_ids):
             errors.append(f"{prefix} _image_name_to_id values do not match _image_ids")

        # --- Cross-Reference Checks ---
        if self._num_images > 0 and not np.all(np.isin(self._image_camera_ids, self._camera_ids)):
            invalid_cam_ids = self._image_camera_ids[~np.isin(self._image_camera_ids, self._camera_ids)]
            errors.append(f"{prefix} _image_camera_ids contains invalid camera IDs (e.g., {invalid_cam_ids[0]})")

        # --- Feature Consistency ---
        num_total_features = len(self._all_xys)
        if num_total_features != len(self._all_point3D_ids):
            errors.append(f"{prefix} len(_all_xys) ({num_total_features}) != len(_all_point3D_ids) ({len(self._all_point3D_ids)})")
        if self._num_images > 0:
            if np.any(self._image_feature_indices[:, 0] > self._image_feature_indices[:, 1]):
                errors.append(f"{prefix} _image_feature_indices contains start > end")
            max_feat_idx = np.max(self._image_feature_indices[:, 1]) if self._num_images > 0 else 0
            if max_feat_idx > num_total_features:
                errors.append(f"{prefix} Max index in _image_feature_indices ({max_feat_idx}) exceeds feature array length ({num_total_features})")

        # Check that valid point IDs in features exist in points3D
        valid_feature_p3d_ids = self._all_point3D_ids[self._all_point3D_ids != INVALID_POINT3D_ID]
        if len(valid_feature_p3d_ids) > 0 and not np.all(np.isin(valid_feature_p3d_ids, self._point_ids)):
             invalid_p3d_refs = valid_feature_p3d_ids[~np.isin(valid_feature_p3d_ids, self._point_ids)]
             errors.append(f"{prefix} _all_point3D_ids references non-existent Point3D IDs (e.g., {invalid_p3d_refs[0]})")

        # --- Track Consistency ---
        num_total_tracks = len(self._all_track_image_ids)
        if num_total_tracks != len(self._all_track_point2D_idxs):
            errors.append(f"{prefix} len(_all_track_image_ids) ({num_total_tracks}) != len(_all_track_point2D_idxs) ({len(self._all_track_point2D_idxs)})")
        if self._num_points3D > 0:
            if np.any(self._point_track_indices[:, 0] > self._point_track_indices[:, 1]):
                errors.append(f"{prefix} _point_track_indices contains start > end")
            max_track_idx = np.max(self._point_track_indices[:, 1]) if self._num_points3D > 0 else 0
            if max_track_idx > num_total_tracks:
                errors.append(f"{prefix} Max index in _point_track_indices ({max_track_idx}) exceeds track array length ({num_total_tracks})")

        # Check that image IDs in tracks exist
        if num_total_tracks > 0 and not np.all(np.isin(self._all_track_image_ids, self._image_ids)):
            invalid_track_img_ids = self._all_track_image_ids[~np.isin(self._all_track_image_ids, self._image_ids)]
            errors.append(f"{prefix} _all_track_image_ids references non-existent Image IDs (e.g., {invalid_track_img_ids[0]})")

        # --- Bi-directional Track Check (Iterative) ---
        # This is the most complex check
        for p3d_row, p3d_id in enumerate(self._point_ids):
            track_start, track_end = self._point_track_indices[p3d_row]
            if track_start >= track_end: continue # Skip empty tracks

            # Iterate through this point's track entries
            for track_idx in range(track_start, track_end):
                img_id = self._all_track_image_ids[track_idx]
                p2d_idx = self._all_track_point2D_idxs[track_idx]

                # Check if image ID from track exists (should have been caught above, but double-check)
                if img_id not in self._image_id_to_row:
                    errors.append(f"{prefix} Point3D {p3d_id} track references deleted/invalid Image ID {img_id} at track index {track_idx}")
                    continue

                img_row = self._image_id_to_row[img_id]
                img_feat_start, img_feat_end = self._image_feature_indices[img_row]
                num_features_in_image = img_feat_end - img_feat_start

                # Check if point2D index is valid for the image
                if not (0 <= p2d_idx < num_features_in_image):
                    errors.append(f"{prefix} Point3D {p3d_id} track references out-of-bounds point2D index {p2d_idx} for image {img_id} (size {num_features_in_image}) at track index {track_idx}")
                    continue

                # Calculate the global index into _all_xys and _all_point3D_ids
                global_feat_idx = img_feat_start + p2d_idx

                # Check bounds for the global index (should be okay if feature ranges are correct)
                if not (0 <= global_feat_idx < len(self._all_point3D_ids)):
                     errors.append(f"{prefix} Point3D {p3d_id} track leads to invalid global feature index {global_feat_idx} (image {img_id}, p2d_idx {p2d_idx})")
                     continue

                # The core check: Does the feature point back to this 3D point?
                feature_p3d_id = self._all_point3D_ids[global_feat_idx]
                if feature_p3d_id != p3d_id:
                    errors.append(f"{prefix} Point3D {p3d_id} track inconsistency: Image {img_id} feature at index {p2d_idx} (global {global_feat_idx}) points to Point3D {feature_p3d_id} instead.")

        return errors

