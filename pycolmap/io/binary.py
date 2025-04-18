import os
import struct
import numpy as np
import concurrent.futures
from typing import Tuple, BinaryIO, List

from .data import CameraData, ImageData, Point3DData
from ..types import CAMERA_MODEL_IDS, INVALID_POINT3D_ID, MAX_CAMERA_PARAMS

def _read_next_bytes(fid: BinaryIO, num_bytes: int, format_char_sequence: str, endian: str = "<") -> tuple:
    """Read and unpack the next bytes from a binary file.
    
    Args:
        fid: Open file object
        num_bytes: Number of bytes to read
        format_char_sequence: Format characters for struct.unpack
        endian_character: Endian character ('<' for little endian)
        
    Returns:
        Tuple of unpacked values
    """
    data = fid.read(num_bytes)

    if len(data) != num_bytes:
        raise EOFError(f"Could not read {num_bytes} bytes. File truncated?")

    return struct.unpack(endian + format_char_sequence, data)


# --- Modified Read Functions ---

def read_cameras_binary(path: str) -> CameraData:
    """Read camera parameters from a COLMAP binary file.
    
    Args:
        path: Path to the cameras.bin file
        
    Returns:
        CameraData, packed object with several camera data
    """
    camera_data = CameraData()

    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        camera_data.num_cameras = num_cameras

        if num_cameras == 0:
            camera_data.ids = np.empty(0, dtype=np.uint16)
            camera_data.model_ids = np.empty(0, dtype=np.uint8)
            camera_data.widths = np.empty(0, dtype=np.uint32)
            camera_data.heights = np.empty(0, dtype=np.uint32)
            camera_data.params = np.empty((0, MAX_CAMERA_PARAMS), dtype=np.float32)
            return camera_data

        # Pre-allocate NumPy arrays
        ids_arr = np.empty(num_cameras, dtype=np.uint16)
        model_ids_arr = np.empty(num_cameras, dtype=np.uint8) # Assuming <= 255 models
        widths_arr = np.empty(num_cameras, dtype=np.uint32)
        heights_arr = np.empty(num_cameras, dtype=np.uint32)
        params_arr = np.empty((num_cameras, MAX_CAMERA_PARAMS), dtype=np.float32)

        for i in range(num_cameras):
            cam_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")

            ids_arr[i] = cam_id
            model_ids_arr[i] = model_id
            widths_arr[i] = width
            heights_arr[i] = height

            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)

            # Pad parameters to MAX_CAMERA_PARAMS and assign directly
            params_arr[i, :num_params] = params
            params_arr[i, num_params:] = 0 # Ensure padding is zero

    # Assign pre-allocated arrays directly
    camera_data.ids = ids_arr
    camera_data.model_ids = model_ids_arr
    camera_data.widths = widths_arr
    camera_data.heights = heights_arr
    camera_data.params = params_arr

    return camera_data


def read_images_binary(path: str, only_3d_features: bool) -> ImageData:
    """Read image data from a COLMAP binary file.
    
    Args:
        path: Path to the images.bin file
        
    Returns:
        ImageData object containing image data.
    """
    image_data = ImageData()
    names_list = [] # Keep as list as names have variable length
    all_xys_list = [] # Keep as list as total size unknown
    all_point3D_ids_list = [] # Keep as list as total size unknown
    current_feature_idx = 0

    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        image_data.num_images = num_reg_images

        if num_reg_images >= 65_535:
            raise ValueError("Number of images exceeds 65535")

        if num_reg_images == 0:
            # Handle empty case gracefully
            image_data.ids = np.empty(0, dtype=np.uint16)
            image_data.qvecs = np.empty((0, 4), dtype=np.float64)
            image_data.tvecs = np.empty((0, 3), dtype=np.float64)
            image_data.camera_ids = np.empty(0, dtype=np.uint16)
            image_data.names = []
            image_data.all_xys = np.empty((0, 2), dtype=np.float64)
            image_data.all_point3D_ids = np.empty(0, dtype=np.uint32)
            image_data.feature_indices = np.empty((0, 2), dtype=np.uint32)
            return image_data

        # Pre-allocate arrays where size is known
        ids_arr = np.empty(num_reg_images, dtype=np.uint16)
        qvecs_arr = np.empty((num_reg_images, 4), dtype=np.float64)
        tvecs_arr = np.empty((num_reg_images, 3), dtype=np.float64)
        camera_ids_arr = np.empty(num_reg_images, dtype=np.uint16)
        feature_indices_arr = np.empty((num_reg_images, 2), dtype=np.uint32)

        for i in range(num_reg_images):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read_next_bytes(fid, 64, "idddddddi")

            ids_arr[i] = img_id
            qvecs_arr[i] = (qw, qx, qy, qz)
            tvecs_arr[i] = (tx, ty, tz)
            camera_ids_arr[i] = cam_id

            # Read image name
            name = ""
            char_byte = _read_next_bytes(fid, 1, "c")[0]
            while char_byte != b"\x00":
                name += char_byte.decode("utf-8")
                char_byte = _read_next_bytes(fid, 1, "c")[0]
            names_list.append(name)

            # Read points
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            start_idx = current_feature_idx
            num_valid_points_in_image = 0

            # Parse points
            if num_points2D > 0:
                x_y_id_s = _read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)

                for j in range(num_points2D):
                    x = x_y_id_s[3 * j]
                    y = x_y_id_s[3 * j + 1]
                    p3d_id = x_y_id_s[3 * j + 2]

                    if only_3d_features and p3d_id == INVALID_POINT3D_ID:
                        continue

                    all_xys_list.append((x, y))
                    all_point3D_ids_list.append(p3d_id)
                    num_valid_points_in_image += 1

            current_feature_idx += num_valid_points_in_image
            feature_indices_arr[i] = (start_idx, current_feature_idx)

    # Assign pre-allocated arrays
    image_data.ids = ids_arr
    image_data.qvecs = qvecs_arr
    image_data.tvecs = tvecs_arr
    image_data.camera_ids = camera_ids_arr
    image_data.feature_indices = feature_indices_arr

    # Assign names list directly
    image_data.names = names_list

    # Convert lists to arrays
    image_data.all_xys = np.array(all_xys_list, dtype=np.float64)
    image_data.all_point3D_ids = np.array(all_point3D_ids_list, dtype=np.uint32)

    # Explicitly delete temporary lists to potentially free memory sooner
    del all_xys_list
    del all_point3D_ids_list

    return image_data


def read_points3D_binary(path: str) -> Point3DData:
    """Read 3D points from a COLMAP binary file.
    
    Args:
        path: Path to the points3D.bin file
        
    Returns:
        Point3DData object containing 3D point data.
    """
    point_data = Point3DData()
    all_track_img_ids_list = [] # Keep as list as total size unknown
    all_track_p2d_idxs_list = [] # Keep as list as total size unknown
    current_track_idx = 0

    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        point_data.num_points = num_points

        if num_points == 0:
            # Handle empty case gracefully
            point_data.ids = np.empty(0, dtype=np.uint32)
            point_data.xyzs = np.empty((0, 3), dtype=np.float64)
            point_data.rgbs = np.empty((0, 3), dtype=np.uint8)
            point_data.errors = np.empty(0, dtype=np.float64)
            point_data.all_track_image_ids = np.empty(0, dtype=np.uint16)
            point_data.all_track_point2D_idxs = np.empty(0, dtype=np.uint32)
            point_data.track_indices = np.empty((0, 2), dtype=np.uint32)
            return point_data

        # Pre-allocate arrays where size is known
        ids_arr = np.empty(num_points, dtype=np.uint32)
        xyzs_arr = np.empty((num_points, 3), dtype=np.float64)
        rgbs_arr = np.empty((num_points, 3), dtype=np.uint8)
        errors_arr = np.empty(num_points, dtype=np.float64)
        track_indices_arr = np.empty((num_points, 2), dtype=np.uint32)

        for i in range(num_points):
            p3d_id, x, y, z, r, g, b, error = _read_next_bytes(fid, 43, "qdddBBBd")
            ids_arr[i] = p3d_id
            xyzs_arr[i] = (x, y, z)
            rgbs_arr[i] = (r, g, b)
            errors_arr[i] = error

            track_len = _read_next_bytes(fid, 8, "Q")[0]
            start_idx = current_track_idx

            if track_len > 0:
                track_elems = _read_next_bytes(fid, 8 * track_len, "ii" * track_len)
                for j in range(track_len):
                    all_track_img_ids_list.append(track_elems[2 * j])
                    all_track_p2d_idxs_list.append(track_elems[2 * j + 1])

            current_track_idx += track_len
            track_indices_arr[i] = (start_idx, current_track_idx)

    # Assign pre-allocated arrays
    point_data.ids = ids_arr
    point_data.xyzs = xyzs_arr
    point_data.rgbs = rgbs_arr
    point_data.errors = errors_arr
    point_data.track_indices = track_indices_arr

    # Convert lists to arrays
    point_data.all_track_image_ids = np.array(all_track_img_ids_list, dtype=np.uint16)
    point_data.all_track_point2D_idxs = np.array(all_track_p2d_idxs_list, dtype=np.uint32)

    # Explicitly delete temporary lists to potentially free memory sooner
    del all_track_img_ids_list
    del all_track_p2d_idxs_list

    return point_data

# --- Modified Write Functions ---
# These now require the internal numpy arrays from ColmapReconstruction

def write_cameras_binary(
    ids: np.ndarray, model_ids: np.ndarray, widths: np.ndarray,
    heights: np.ndarray, params: np.ndarray, path: str
) -> None:
    """Write camera parameters to a COLMAP binary file.
    
    Args:
        path: Output file path
    """
    num_cameras = len(ids)

    with open(path, "wb") as fid:
        # Write number of cameras
        fid.write(struct.pack("<Q", num_cameras))

        for i in range(num_cameras):

            # Get model ID
            cam_id = ids[i]
            model_id = model_ids[i]
            width = widths[i]
            height = heights[i]
            cam_params = params[i] # This is the padded array

            if model_id not in CAMERA_MODEL_IDS:
                 raise ValueError(f"Invalid model ID {model_id} found for camera {cam_id}")
            num_params = CAMERA_MODEL_IDS[model_id].num_params

            fid.write(struct.pack("<i", cam_id))
            fid.write(struct.pack("<i", model_id))
            fid.write(struct.pack("<Q", width))
            fid.write(struct.pack("<Q", height))
            # Write only the non-padded part
            fid.write(struct.pack(f"<{num_params}d", *cam_params[:num_params]))

def write_images_binary(
    ids: np.ndarray, qvecs: np.ndarray, tvecs: np.ndarray, camera_ids: np.ndarray,
    names: List[str], all_xys: np.ndarray, all_point3D_ids: np.ndarray,
    feature_indices: np.ndarray, path: str
) -> None:
    """Write image data to a COLMAP binary file.
    
    Args:
        images: Dictionary of Image objects
        path: Output file path
    """
    num_images = len(ids)

    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", num_images))
        for i in range(num_images):
            fid.write(struct.pack("<i", ids[i]))
            fid.write(struct.pack("<dddd", *qvecs[i]))
            fid.write(struct.pack("<ddd", *tvecs[i]))
            fid.write(struct.pack("<i", camera_ids[i]))

            name_bytes = names[i].encode("utf-8")
            fid.write(struct.pack(f"<{len(name_bytes)}s", name_bytes))
            fid.write(struct.pack("<c", b"\x00")) # Null terminator

            start, end = feature_indices[i]
            num_points2D = end - start
            fid.write(struct.pack("<Q", num_points2D))

            if num_points2D > 0:
                xys = all_xys[start:end]
                p3d_ids = all_point3D_ids[start:end]
                for xy, p3d_id in zip(xys, p3d_ids):
                    fid.write(struct.pack("<dd", xy[0], xy[1]))
                    fid.write(struct.pack("<q", p3d_id))

def write_points3D_binary(
    ids: np.ndarray, xyzs: np.ndarray, rgbs: np.ndarray, errors: np.ndarray,
    all_track_image_ids: np.ndarray, all_track_point2D_idxs: np.ndarray,
    track_indices: np.ndarray, path: str
) -> None:
    """Write 3D points to a COLMAP binary file.
    
    Args:
        points3D: Dictionary of Point3D objects
        path: Output file path
    """
    num_points = len(ids)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", num_points))
        for i in range(num_points):
            fid.write(struct.pack("<Q", ids[i]))
            fid.write(struct.pack("<ddd", *xyzs[i]))
            fid.write(struct.pack("<BBB", *rgbs[i]))
            fid.write(struct.pack("<d", errors[i]))

            start, end = track_indices[i]
            track_len = end - start
            fid.write(struct.pack("<Q", track_len))

            if track_len > 0:
                img_ids = all_track_image_ids[start:end]
                p2d_idxs = all_track_point2D_idxs[start:end]
                for img_id, p2d_idx in zip(img_ids, p2d_idxs):
                    fid.write(struct.pack("<ii", img_id, p2d_idx))


# --- Top-Level Read/Write for Binary ---

def read_binary_model(path: str, only_3d_features: bool) -> Tuple[CameraData, ImageData, Point3DData]:
    """Read a COLMAP binary model from a directory.
    
    Args:
        path: Directory containing the binary model files
        
    Returns:
        Tuple of (cameras, images, points3D) buffer objects
    """
    cameras_path = os.path.join(path, "cameras.bin")
    images_path = os.path.join(path, "images.bin")
    points3D_path = os.path.join(path, "points3D.bin")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_cameras = executor.submit(read_cameras_binary, cameras_path)
        future_images = executor.submit(read_images_binary, images_path, only_3d_features)
        future_points3D = executor.submit(read_points3D_binary, points3D_path)

        try:
            cameras = future_cameras.result()
            images = future_images.result()
            points3D = future_points3D.result()
        except Exception as e:
            print(f"Error reading binary model files: {e}")
            raise e

    return cameras, images, points3D


def write_binary_model(
    cameras: CameraData, images: ImageData, points3D: Point3DData, path: str
) -> None:
    """Writes the internal data structures to a binary model."""
    if not os.path.exists(path):
        os.makedirs(path)

    write_cameras_binary(
        cameras.ids, cameras.model_ids, cameras.widths, cameras.heights, cameras.params,
        os.path.join(path, "cameras.bin")
    )
    write_images_binary(
        images.ids, images.qvecs, images.tvecs, images.camera_ids, images.names,
        images.all_xys, images.all_point3D_ids, images.feature_indices,
        os.path.join(path, "images.bin")
    )
    write_points3D_binary(
        points3D.ids, points3D.xyzs, points3D.rgbs, points3D.errors,
        points3D.all_track_image_ids, points3D.all_track_point2D_idxs, points3D.track_indices,
        os.path.join(path, "points3D.bin")
    )

