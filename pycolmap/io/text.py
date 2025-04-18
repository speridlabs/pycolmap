import os
import numpy as np
from typing import Tuple, List

from .data import CameraData, ImageData, Point3DData
from ..types import CAMERA_MODEL_IDS, INVALID_POINT3D_ID, MAX_CAMERA_PARAMS


def read_cameras_text(path: str) -> CameraData:
    """Read camera parameters from a COLMAP text file.

    Args:
        path: Path to the cameras.txt file

    Returns:
        CameraData object containing camera data.
    """
    camera_data = CameraData()
    num_cameras = 0
    # First pass: Count valid cameras
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                elems = line.split()
                if len(elems) < 4: continue # Basic check
                model_name = elems[1]
                params_count = len(elems) - 4
                # Find model ID and check param count without storing
                model_id = -1
                for id_val, model_info in CAMERA_MODEL_IDS.items():
                    if model_info.model_name == model_name:
                        model_id = id_val
                        break
                if model_id == -1: continue # Unknown model
                if params_count != CAMERA_MODEL_IDS[model_id].num_params: continue # Wrong param count
                # Attempt basic type conversions to catch errors early
                int(elems[0])
                int(elems[2])
                int(elems[3])
                for p in elems[4:]: float(p)
                # If all checks pass, increment count
                num_cameras += 1
            except (ValueError, IndexError):
                continue # Skip malformed lines


    camera_data.num_cameras = num_cameras

    if num_cameras == 0:
        # Handle empty case gracefully
        camera_data.ids = np.empty(0, dtype=np.uint16)
        camera_data.model_ids = np.empty(0, dtype=np.uint8)
        camera_data.widths = np.empty(0, dtype=np.uint32)
        camera_data.heights = np.empty(0, dtype=np.uint32)
        camera_data.params = np.empty((0, MAX_CAMERA_PARAMS), dtype=np.float32)
        return camera_data

    # Pre-allocate NumPy arrays
    ids_arr = np.empty(num_cameras, dtype=np.uint16)
    model_ids_arr = np.empty(num_cameras, dtype=np.uint8)
    widths_arr = np.empty(num_cameras, dtype=np.uint32)
    heights_arr = np.empty(num_cameras, dtype=np.uint32)
    params_arr = np.zeros((num_cameras, MAX_CAMERA_PARAMS), dtype=np.float32) # Initialize with zeros

    # Second pass: Populate arrays
    current_idx = 0
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                elems = line.split()
                if len(elems) < 4: continue # Basic check

                model_name = elems[1]
                params = [float(p) for p in elems[4:]]

                # Find model ID
                model_id = -1
                for id_val, model_info in CAMERA_MODEL_IDS.items():
                    if model_info.model_name == model_name:
                        model_id = id_val
                        break
                if model_id == -1: continue # Unknown model

                # Validate number of parameters again (safety check)
                if len(params) != CAMERA_MODEL_IDS[model_id].num_params:
                    continue # Skip if param count mismatch

                camera_id = int(elems[0])
                width = int(elems[2])
                height = int(elems[3])

                # Populate arrays directly
                ids_arr[current_idx] = camera_id
                model_ids_arr[current_idx] = model_id
                widths_arr[current_idx] = width
                heights_arr[current_idx] = height
                params_arr[current_idx, :len(params)] = params
                current_idx += 1

            except (ValueError, IndexError):
                # This should ideally not happen if the first pass was accurate,
                # but keep it as a safeguard.
                # print(f"Warning: Skipping malformed camera line during second pass: {line}")
                continue # Skip to the next line

    # Assign populated arrays
    camera_data.ids = ids_arr
    camera_data.model_ids = model_ids_arr
    camera_data.widths = widths_arr
    camera_data.heights = heights_arr
    camera_data.params = params_arr

    return camera_data


def read_images_text(path: str, only_3d_features: bool) -> ImageData:
    """Read image data from a COLMAP text file.

    Args:
        path: Path to the images.txt file
        only_3d_features: If True, only store 2D points that have a valid 3D point ID.

    Returns:
        ImageData object containing image data.
    """
    image_data = ImageData()
    num_images = 0

    # First pass: Count valid image entries (pairs of lines)
    with open(path, "r") as fid:
        is_expecting_points = False
        for line in fid:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if is_expecting_points:
                # This line should be points. We don't need to parse it fully here,
                # just validate the previous line was okay and count this pair.
                num_images += 1
                is_expecting_points = False # Reset for the next potential image line
            else:
                # This line should be image properties. Try a basic parse.
                elems = line.split()

                if len(elems) >= 10:
                    try:
                        # Attempt basic conversions to catch errors
                        int(elems[0]) # id
                        float(elems[1]); float(elems[2]); float(elems[3]); float(elems[4]) # qvec
                        float(elems[5]); float(elems[6]); float(elems[7]) # tvec
                        int(elems[8]) # cam_id
                        # elems[9] is name, no conversion needed
                        is_expecting_points = True # If successful, expect points next
                    except (ValueError, IndexError):
                        is_expecting_points = False # Malformed line, reset
                else:
                    is_expecting_points = False # Not enough elements, reset


    image_data.num_images = num_images

    if num_images == 0:
        # Handle empty case gracefully (same as before)
        image_data.ids = np.empty(0, dtype=np.uint16)
        image_data.qvecs = np.empty((0, 4), dtype=np.float64)
        image_data.tvecs = np.empty((0, 3), dtype=np.float64)
        image_data.camera_ids = np.empty(0, dtype=np.uint16)
        image_data.names = []
        image_data.all_xys = np.empty((0, 2), dtype=np.float64)
        image_data.all_point3D_ids = np.empty(0, dtype=np.uint32)
        image_data.feature_indices = np.empty((0, 2), dtype=np.uint32)
        return image_data

    # Pre-allocate NumPy arrays for per-image data
    ids_arr = np.empty(num_images, dtype=np.uint16)
    qvecs_arr = np.empty((num_images, 4), dtype=np.float64)
    tvecs_arr = np.empty((num_images, 3), dtype=np.float64)
    camera_ids_arr = np.empty(num_images, dtype=np.uint16)
    feature_indices_arr = np.empty((num_images, 2), dtype=np.uint32)
    names_list: List[str] = [""] * num_images # Pre-allocate list for names

    # Lists for concatenated feature data (size unknown until second pass)
    all_xys_list: List[Tuple[float, float]] = []
    all_point3D_ids_list: List[int] = []
    current_feature_idx = 0
    current_image_array_idx = 0

    # Second pass: Populate arrays and lists
    with open(path, "r") as fid:
        read_points = False
        # Temp variables for the current image being processed before assigning to arrays
        current_image_id: int = -1
        current_qvec: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        current_tvec: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        current_camera_id: int = -1
        current_image_name: str = ""

        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if read_points:
                # Read points (second line of image data)
                elems = line.split()
                start_idx = current_feature_idx
                num_valid_points_in_image = 0

                if len(elems) > 0 and len(elems) % 3 == 0:
                    for i in range(0, len(elems), 3):
                        try:
                            x = float(elems[i])
                            y = float(elems[i+1])
                            point3D_id = int(elems[i+2])

                            if only_3d_features and point3D_id == INVALID_POINT3D_ID:
                                continue

                            all_xys_list.append((x, y))
                            all_point3D_ids_list.append(point3D_id)
                            num_valid_points_in_image += 1
                        except (ValueError, IndexError):
                            continue # Skip malformed point triplet

                # Store data for the successfully read image in pre-allocated arrays
                if current_image_id != -1: # Check if the previous line was valid
                    ids_arr[current_image_array_idx] = current_image_id
                    qvecs_arr[current_image_array_idx] = current_qvec
                    tvecs_arr[current_image_array_idx] = current_tvec
                    camera_ids_arr[current_image_array_idx] = current_camera_id
                    names_list[current_image_array_idx] = current_image_name # Assign to list
                    current_feature_idx += num_valid_points_in_image
                    feature_indices_arr[current_image_array_idx] = (start_idx, current_feature_idx)
                    current_image_array_idx += 1

                # Reset for next image pair
                read_points = False
                current_image_id = -1 # Mark as processed or invalid

            else:
                # Read image properties (first line of image data)
                elems = line.split()
                current_image_id = -1 # Reset before trying to parse

                if len(elems) >= 10:
                    try:
                        current_image_id = int(elems[0])
                        current_qvec = (float(elems[1]), float(elems[2]), float(elems[3]), float(elems[4]))
                        current_tvec = (float(elems[5]), float(elems[6]), float(elems[7]))
                        current_camera_id = int(elems[8])
                        current_image_name = elems[9]
                        read_points = True # Expect points next
                    except (ValueError, IndexError):
                        read_points = False
                        current_image_id = -1 # Ensure invalid state
                        continue # Skip malformed line
                else:
                    read_points = False
                    current_image_id = -1 # Ensure invalid state
                    continue # Skip line with not enough elements

    # Assign pre-allocated arrays
    image_data.ids = ids_arr
    image_data.qvecs = qvecs_arr
    image_data.tvecs = tvecs_arr
    image_data.camera_ids = camera_ids_arr
    image_data.feature_indices = feature_indices_arr
    image_data.names = names_list # Assign the populated list

    # Convert feature lists to arrays
    image_data.all_xys = np.array(all_xys_list, dtype=np.float64)
    image_data.all_point3D_ids = np.array(all_point3D_ids_list, dtype=np.uint32)

    # Explicitly delete temporary lists

    del all_xys_list, all_point3D_ids_list

    return image_data


def read_points3D_text(path: str) -> Point3DData:
    """Read 3D points from a COLMAP text file.

    Args:
        path: Path to the points3D.txt file

    Returns:
        Point3DData object containing 3D point data.
    """
    point_data = Point3DData()
    num_points = 0

    # First pass: Count valid point entries
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            elems = line.split()
            # Basic check for minimum elements and that track length is even
            if len(elems) >= 8 and (len(elems) - 8) % 2 == 0:
                try:
                    # Attempt basic conversions
                    int(elems[0]) # id
                    float(elems[1]); float(elems[2]); float(elems[3]) # xyz
                    int(elems[4]); int(elems[5]); int(elems[6]) # rgb
                    float(elems[7]) # error
                    # Check track elements if any
                    for i in range(8, len(elems), 2):
                        int(elems[i]) # image_id
                        int(elems[i+1]) # point2d_idx
                    num_points += 1 # Count if all checks pass
                except (ValueError, IndexError):
                    continue # Skip malformed line
            else:
                continue # Skip line with wrong number of elements


    point_data.num_points = num_points

    if num_points == 0:
        # Handle empty case gracefully (same as before)
        point_data.ids = np.empty(0, dtype=np.uint32)
        point_data.xyzs = np.empty((0, 3), dtype=np.float64)
        point_data.rgbs = np.empty((0, 3), dtype=np.uint8)
        point_data.errors = np.empty(0, dtype=np.float64)
        point_data.all_track_image_ids = np.empty(0, dtype=np.uint16)
        point_data.all_track_point2D_idxs = np.empty(0, dtype=np.uint32)
        point_data.track_indices = np.empty((0, 2), dtype=np.uint32)
        return point_data

    # Pre-allocate NumPy arrays for per-point data
    ids_arr = np.empty(num_points, dtype=np.uint32)
    xyzs_arr = np.empty((num_points, 3), dtype=np.float64)
    rgbs_arr = np.empty((num_points, 3), dtype=np.uint8)
    errors_arr = np.empty(num_points, dtype=np.float64)
    track_indices_arr = np.empty((num_points, 2), dtype=np.uint32)

    # Lists for concatenated track data (size unknown until second pass)
    all_track_img_ids_list: List[int] = []
    all_track_p2d_idxs_list: List[int] = []
    current_track_idx = 0
    current_point_array_idx = 0

    # Second pass: Populate arrays and lists
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            elems = line.split()
            # Repeat validation checks from first pass for robustness
            if len(elems) >= 8 and (len(elems) - 8) % 2 == 0:
                try:
                    point3D_id = int(elems[0])
                    xyz = (float(elems[1]), float(elems[2]), float(elems[3]))
                    rgb = (int(elems[4]), int(elems[5]), int(elems[6]))
                    error = float(elems[7])

                    # Populate pre-allocated arrays
                    ids_arr[current_point_array_idx] = point3D_id
                    xyzs_arr[current_point_array_idx] = xyz
                    rgbs_arr[current_point_array_idx] = rgb
                    errors_arr[current_point_array_idx] = error

                    # Parse track and append to lists
                    start_idx = current_track_idx
                    num_track_elems = 0
                    for i in range(8, len(elems), 2):
                        try:
                            image_id = int(elems[i])
                            point2D_idx = int(elems[i+1])
                            all_track_img_ids_list.append(image_id)
                            all_track_p2d_idxs_list.append(point2D_idx)
                            num_track_elems += 1
                        except (ValueError, IndexError):
                            # Skip malformed track pair but continue with the point
                            continue

                    current_track_idx += num_track_elems
                    track_indices_arr[current_point_array_idx] = (start_idx, current_track_idx)
                    current_point_array_idx += 1

                except (ValueError, IndexError):
                    # This should ideally not happen if first pass was accurate
                    continue # Skip malformed line
            else:
                continue # Skip line with wrong number of elements

    # Assign pre-allocated arrays
    point_data.ids = ids_arr
    point_data.xyzs = xyzs_arr
    point_data.rgbs = rgbs_arr
    point_data.errors = errors_arr
    point_data.track_indices = track_indices_arr

    # Convert track lists to arrays
    point_data.all_track_image_ids = np.array(all_track_img_ids_list, dtype=np.uint16)
    point_data.all_track_point2D_idxs = np.array(all_track_p2d_idxs_list, dtype=np.uint32)

    # Explicitly delete temporary lists

    del all_track_img_ids_list, all_track_p2d_idxs_list

    return point_data


def write_cameras_text(cameras: CameraData, path: str) -> None:
    """Write camera parameters to a COLMAP text file.

    Args:
        cameras: CameraData object
        path: Output file path
    """
    with open(path, "w") as fid:
        # Write header
        fid.write("# Camera list with one line of data per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fid.write(f"# Number of cameras: {cameras.num_cameras}\n")

        # Write cameras
        # Create a mapping from ID to index for potentially unsorted IDs
        id_to_idx = {cam_id: i for i, cam_id in enumerate(cameras.ids)}
        sorted_ids = sorted(cameras.ids)

        for camera_id in sorted_ids:
            idx = id_to_idx[camera_id]
            model_id = cameras.model_ids[idx]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = cameras.widths[idx]
            height = cameras.heights[idx]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = cameras.params[idx, :num_params] # Get only the relevant params
            params_str = " ".join(map(str, params))
            fid.write(f"{camera_id} {model_name} {width} {height} {params_str}\n")


def write_images_text(images: ImageData, path: str) -> None:
    """Write image data to a COLMAP text file.

    Args:
        images: ImageData object
        path: Output file path
    """
    with open(path, "w") as fid:
        # Write header
        fid.write("# Image list with two lines of data per image:\n")
        fid.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fid.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fid.write(f"# Number of images: {images.num_images}\n")

        # Write images
        # Create a mapping from ID to index for potentially unsorted IDs
        id_to_idx = {img_id: i for i, img_id in enumerate(images.ids)}
        sorted_ids = sorted(images.ids)

        for image_id in sorted_ids:
            idx = id_to_idx[image_id]
            qvec = images.qvecs[idx]
            tvec = images.tvecs[idx]
            camera_id = images.camera_ids[idx]
            name = images.names[idx] # Assumes names list is ordered corresponding to ids array

            # Write image properties
            qvec_str = " ".join(map(str, qvec))
            tvec_str = " ".join(map(str, tvec))
            fid.write(f"{image_id} {qvec_str} {tvec_str} {camera_id} {name}\n")

            # Write points
            start, end = images.feature_indices[idx]
            points_str = ""
            if start < end: # Check if there are points for this image
                xys = images.all_xys[start:end]
                point3D_ids = images.all_point3D_ids[start:end]
                for xy, point3D_id in zip(xys, point3D_ids):
                    # Use INVALID_POINT3D_ID (-1) for points without a 3D correspondence in text format
                    p3d_id_text = point3D_id if point3D_id != INVALID_POINT3D_ID else -1
                    points_str += f"{xy[0]} {xy[1]} {p3d_id_text} "

            # Write points line (even if empty)
            fid.write(f"{points_str.rstrip()}\n") # Use rstrip to remove trailing space


def write_points3D_text(points3D: Point3DData, path: str) -> None:
    """Write 3D points to a COLMAP text file.

    Args:
        points3D: Point3DData object
        path: Output file path
    """
    with open(path, "w") as fid:
        # Write header
        fid.write("# 3D point list with one line of data per point:\n")
        fid.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fid.write(f"# Number of points: {points3D.num_points}\n")

        # Write points
        # Create a mapping from ID to index for potentially unsorted IDs
        id_to_idx = {p3d_id: i for i, p3d_id in enumerate(points3D.ids)}
        sorted_ids = sorted(points3D.ids)

        for point3D_id in sorted_ids:
            idx = id_to_idx[point3D_id]
            xyz = points3D.xyzs[idx]
            rgb = points3D.rgbs[idx]
            error = points3D.errors[idx]

            # Write point properties
            xyz_str = " ".join(map(str, xyz))
            rgb_str = " ".join(map(str, rgb))

            # Write track
            start, end = points3D.track_indices[idx]
            track_str = ""
            if start < end: # Check if there are track elements
                image_ids = points3D.all_track_image_ids[start:end]
                point2D_idxs = points3D.all_track_point2D_idxs[start:end]
                for img_id, p2D_idx in zip(image_ids, point2D_idxs):
                    track_str += f"{img_id} {p2D_idx} "

            # Write point line
            fid.write(f"{point3D_id} {xyz_str} {rgb_str} {error} {track_str.rstrip()}\n") # Use rstrip


def read_text_model(path: str, only_3d_features: bool) -> Tuple[CameraData, ImageData, Point3DData]:
    """Read a COLMAP text model from a directory.


    Args:
        path: Directory containing the text model files (cameras.txt, images.txt, points3D.txt)
        only_3d_features: If True, only store 2D points in images.txt that have a valid 3D point ID.

    Returns:
        Tuple of (CameraData, ImageData, Point3DData) objects.
    """
    cameras = read_cameras_text(os.path.join(path, "cameras.txt"))
    # Pass the flag down to read_images_text
    images = read_images_text(os.path.join(path, "images.txt"), only_3d_features)
    points3D = read_points3D_text(os.path.join(path, "points3D.txt"))

    return cameras, images, points3D


def write_text_model(cameras: CameraData, images: ImageData,
                    points3D: Point3DData, path: str) -> None:
    """Write a COLMAP text model to a directory.

    Args:
        cameras: CameraData object
        images: ImageData object
        points3D: Point3DData object
        path: Output directory
    """
    if not os.path.exists(path):
        os.makedirs(path)

    write_cameras_text(cameras, os.path.join(path, "cameras.txt"))
    write_images_text(images, os.path.join(path, "images.txt"))
    write_points3D_text(points3D, os.path.join(path, "points3D.txt"))
