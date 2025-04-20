import os
import gc
import struct
import functools
import numpy as np
import concurrent.futures
from typing import Tuple, BinaryIO, List

from .data import CameraData, ImageData, Point3DData
from ..types import CAMERA_MODEL_IDS, INVALID_POINT3D_ID, MAX_CAMERA_PARAMS

# |-----------------------------------------------------------------------|
# |-------- This file is like this because of memory fragmentation -------|
# |-----------------------------------------------------------------------|

# Struct caching with LRU cache to reduce struct objects
@functools.lru_cache(maxsize=128)
def _get_struct(format_str):
    return struct.Struct(format_str)

def clear_memory_caches():
    """Clear all memory caches to help garbage collection."""

    _get_struct.cache_clear()
    for _ in range(3):
        gc.collect()

def _read_next_bytes(fid: BinaryIO, num_bytes: int, format_char_sequence: str, endian: str = "<") -> tuple:
    """Read and unpack the next bytes from a binary file with cached struct objects."""
    data = fid.read(num_bytes)
    
    if len(data) != num_bytes:
        raise EOFError(f"Could not read {num_bytes} bytes. File truncated?")
    
    # Use cached struct object
    format_string = endian + format_char_sequence
    struct_obj = _get_struct(format_string)
    return struct_obj.unpack(data)

def read_cameras_binary(path: str) -> CameraData:
    camera_data = CameraData()
    
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        camera_data.num_cameras = num_cameras
        
        if num_cameras == 0:
            camera_data.ids = np.empty(0, dtype=np.uint16)
            camera_data.model_ids = np.empty(0, dtype=np.uint8)
            camera_data.widths = np.empty(0, dtype=np.uint32)
            camera_data.heights = np.empty(0, dtype=np.uint32)
            camera_data.params = np.empty((0, MAX_CAMERA_PARAMS), dtype=np.float64)
            return camera_data
        
        # Pre-allocate arrays
        ids_arr = np.empty(num_cameras, dtype=np.uint16)
        model_ids_arr = np.empty(num_cameras, dtype=np.uint8)
        widths_arr = np.empty(num_cameras, dtype=np.uint32)
        heights_arr = np.empty(num_cameras, dtype=np.uint32)
        params_arr = np.empty((num_cameras, MAX_CAMERA_PARAMS), dtype=np.float64)
        
        for i in range(num_cameras):
            cam_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            
            ids_arr[i] = cam_id
            model_ids_arr[i] = model_id
            widths_arr[i] = width
            heights_arr[i] = height
            
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            
            params_arr[i, :num_params] = params
            params_arr[i, num_params:] = 0
    
    # Assign arrays
    camera_data.ids = ids_arr
    camera_data.model_ids = model_ids_arr
    camera_data.widths = widths_arr
    camera_data.heights = heights_arr
    camera_data.params = params_arr
    
    return camera_data

def read_images_binary_optimized(path: str, only_3d_features: bool) -> ImageData:
    """Read image data from a COLMAP binary file with memory optimization."""
    image_data = ImageData()
    
    # First pass: Count valid points to avoid growing lists
    total_valid_points = 0
    num_reg_images = 0
    names_list = []
    file_pos_after_names = []  # Store file positions to avoid re-reading names
    
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
        
        # Count valid points first to avoid growing lists
        for i in range(num_reg_images):
            # Skip image header
            header = _read_next_bytes(fid, 64, "idddddddi")
            
            # Read image name more efficiently
            name_bytes = bytearray()
            while True:
                byte = fid.read(1)
                if byte == b"\x00":
                    break
                name_bytes.extend(byte)
            
            name = name_bytes.decode("utf-8")
            names_list.append(name)
            
            # Save file position after name
            file_pos_after_names.append(fid.tell())
            
            # Read points count
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            
            if num_points2D > 0:
                # Read all points as raw bytes instead of unpacking
                points_data = fid.read(24 * num_points2D)
                
                # Count valid points - we create a view of the raw data
                for j in range(num_points2D):
                    # Extract just the point3D_id at offset 16 bytes within each point
                    p3d_id = struct.unpack_from("<q", points_data, j * 24 + 16)[0]
                    if not only_3d_features or p3d_id != INVALID_POINT3D_ID:
                        total_valid_points += 1
    
    # Now pre-allocate arrays with the exact size needed
    all_xys = np.empty((total_valid_points, 2), dtype=np.float64)
    all_point3D_ids = np.empty(total_valid_points, dtype=np.uint32)
    
    # Pre-allocate other arrays
    ids_arr = np.empty(num_reg_images, dtype=np.uint16)
    qvecs_arr = np.empty((num_reg_images, 4), dtype=np.float64)
    tvecs_arr = np.empty((num_reg_images, 3), dtype=np.float64)
    camera_ids_arr = np.empty(num_reg_images, dtype=np.uint16)
    feature_indices_arr = np.empty((num_reg_images, 2), dtype=np.uint32)
    
    # Second pass to fill arrays
    current_feature_idx = 0
    
    with open(path, "rb") as fid:
        # Skip num_images
        _ = fid.read(8)
        
        for i in range(num_reg_images):
            # Read image header
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read_next_bytes(fid, 64, "idddddddi")
            
            ids_arr[i] = img_id
            qvecs_arr[i] = (qw, qx, qy, qz)
            tvecs_arr[i] = (tx, ty, tz)
            camera_ids_arr[i] = cam_id
            
            # Skip over the name (we already have it)
            while fid.read(1) != b"\x00":
                pass
            
            # Read points
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            start_idx = current_feature_idx
            
            if num_points2D > 0:
                # Read all points as raw bytes
                points_data = fid.read(24 * num_points2D)
                
                # Process without creating large tuples
                for j in range(num_points2D):
                    offset = j * 24
                    x = struct.unpack_from("<d", points_data, offset)[0]
                    y = struct.unpack_from("<d", points_data, offset + 8)[0]
                    p3d_id = struct.unpack_from("<q", points_data, offset + 16)[0]
                    
                    if only_3d_features and p3d_id == INVALID_POINT3D_ID:
                        continue
                    
                    all_xys[current_feature_idx, 0] = x
                    all_xys[current_feature_idx, 1] = y
                    all_point3D_ids[current_feature_idx] = p3d_id
                    current_feature_idx += 1
                
                # Explicitly delete
                del points_data
            
            feature_indices_arr[i] = (start_idx, current_feature_idx)
    
    # Assign arrays
    image_data.ids = ids_arr
    image_data.qvecs = qvecs_arr
    image_data.tvecs = tvecs_arr
    image_data.camera_ids = camera_ids_arr
    image_data.feature_indices = feature_indices_arr
    image_data.names = names_list
    image_data.all_xys = all_xys
    image_data.all_point3D_ids = all_point3D_ids
    
    return image_data

def read_points3D_binary_optimized(path: str) -> Point3DData:
    """Read 3D points from a COLMAP binary file with memory optimization."""
    point_data = Point3DData()
    
    # First pass: Count track elements
    total_track_elements = 0
    
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        point_data.num_points = num_points
        
        if num_points == 0:
            # Handle empty case
            point_data.ids = np.empty(0, dtype=np.uint32)
            point_data.xyzs = np.empty((0, 3), dtype=np.float64)
            point_data.rgbs = np.empty((0, 3), dtype=np.uint8)
            point_data.errors = np.empty(0, dtype=np.float64)
            point_data.all_track_image_ids = np.empty(0, dtype=np.uint16)
            point_data.all_track_point2D_idxs = np.empty(0, dtype=np.uint32)
            point_data.track_indices = np.empty((0, 2), dtype=np.uint32)
            return point_data
        
        # Create a buffer for the track counts to avoid re-reading
        track_lengths = np.empty(num_points, dtype=np.uint64)
        
        for i in range(num_points):
            # Skip point data (43 bytes)
            _ = fid.read(43)
            
            # Read track length
            track_len = _read_next_bytes(fid, 8, "Q")[0]
            track_lengths[i] = track_len
            total_track_elements += track_len
            
            if track_len > 0:
                # Skip track elements (8 bytes each)
                fid.seek(8 * track_len, 1)
    
    # Pre-allocate arrays
    ids_arr = np.empty(num_points, dtype=np.uint32)
    xyzs_arr = np.empty((num_points, 3), dtype=np.float64)
    rgbs_arr = np.empty((num_points, 3), dtype=np.uint8)
    errors_arr = np.empty(num_points, dtype=np.float64)
    track_indices_arr = np.empty((num_points, 2), dtype=np.uint32)
    
    # Allocate track arrays with exact size
    all_track_image_ids = np.empty(total_track_elements, dtype=np.uint16)
    all_track_point2D_idxs = np.empty(total_track_elements, dtype=np.uint32)
    
    # Second pass to fill arrays
    current_track_idx = 0
    
    with open(path, "rb") as fid:
        # Skip num_points
        _ = fid.read(8)
        
        for i in range(num_points):
            p3d_id, x, y, z, r, g, b, error = _read_next_bytes(fid, 43, "qdddBBBd")
            
            ids_arr[i] = p3d_id
            xyzs_arr[i] = (x, y, z)
            rgbs_arr[i] = (r, g, b)
            errors_arr[i] = error
            
            # Read track length
            track_len = int(track_lengths[i])  # Use cached value
            _ = fid.read(8)  # Skip the track length since we already know it
            
            start_idx = current_track_idx
            
            if track_len > 0:
                # Read track elements as raw bytes to avoid large tuples
                track_data = fid.read(8 * track_len)
                
                # Process without creating large tuples
                for j in range(track_len):
                    offset = j * 8
                    img_id = struct.unpack_from("<i", track_data, offset)[0]
                    p2d_idx = struct.unpack_from("<i", track_data, offset + 4)[0]
                    
                    all_track_image_ids[current_track_idx] = img_id
                    all_track_point2D_idxs[current_track_idx] = p2d_idx
                    current_track_idx += 1
                
                # Explicitly delete
                del track_data
            
            track_indices_arr[i] = (start_idx, current_track_idx)
    
    # Assign arrays
    point_data.ids = ids_arr
    point_data.xyzs = xyzs_arr
    point_data.rgbs = rgbs_arr
    point_data.errors = errors_arr
    point_data.track_indices = track_indices_arr
    point_data.all_track_image_ids = all_track_image_ids
    point_data.all_track_point2D_idxs = all_track_point2D_idxs
    
    return point_data

def relocate_to_new_memory(camera_data: CameraData, image_data: ImageData, point_data: Point3DData) -> Tuple[CameraData, ImageData, Point3DData]:
    """
    Create fresh memory allocations to avoid fragmentation.
    """
    # Create brand new objects
    new_camera_data = CameraData()
    new_image_data = ImageData()
    new_point_data = Point3DData()
    
    # Copy camera data
    new_camera_data.num_cameras = camera_data.num_cameras
    new_camera_data.ids = camera_data.ids.copy()
    new_camera_data.model_ids = camera_data.model_ids.copy()
    new_camera_data.widths = camera_data.widths.copy()
    new_camera_data.heights = camera_data.heights.copy()
    new_camera_data.params = camera_data.params.copy()
    
    # Copy image data
    new_image_data.num_images = image_data.num_images
    new_image_data.ids = image_data.ids.copy()
    new_image_data.qvecs = image_data.qvecs.copy()
    new_image_data.tvecs = image_data.tvecs.copy()
    new_image_data.camera_ids = image_data.camera_ids.copy()
    new_image_data.names = list(image_data.names)  # Create a new list
    new_image_data.all_xys = image_data.all_xys.copy()
    new_image_data.all_point3D_ids = image_data.all_point3D_ids.copy()
    new_image_data.feature_indices = image_data.feature_indices.copy()
    
    # Copy point3D data
    new_point_data.num_points = point_data.num_points
    new_point_data.ids = point_data.ids.copy()
    new_point_data.xyzs = point_data.xyzs.copy()
    new_point_data.rgbs = point_data.rgbs.copy()
    new_point_data.errors = point_data.errors.copy()
    new_point_data.all_track_image_ids = point_data.all_track_image_ids.copy()
    new_point_data.all_track_point2D_idxs = point_data.all_track_point2D_idxs.copy()
    new_point_data.track_indices = point_data.track_indices.copy()
    
    # Delete old objects to release memory
    del camera_data
    del image_data
    del point_data
    
    # Force garbage collection
    gc.collect()
    
    return new_camera_data, new_image_data, new_point_data

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
    
    try:
        clear_memory_caches()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks to read files concurrently
            future_cameras = executor.submit(read_cameras_binary, cameras_path)
            future_points3D = executor.submit(read_points3D_binary_optimized, points3D_path)
            future_images = executor.submit(read_images_binary_optimized, images_path, only_3d_features)

            # Retrieve results - .result() blocks until the future is complete
            # and raises any exceptions encountered during execution.
            cameras = future_cameras.result()
            points3D = future_points3D.result()
            images = future_images.result()

        # Relocate memory after all data is loaded and retrieved from threads
        cameras, images, points3D = relocate_to_new_memory(cameras, images, points3D)

        clear_memory_caches()

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

