import os
import struct
import concurrent.futures
from typing import Dict, Tuple, BinaryIO

from ..image import Image
from ..camera import Camera
from ..point3d import Point3D
from ..types import CAMERA_MODEL_NAMES, CAMERA_MODEL_IDS, INVALID_POINT3D_ID

def _read_next_bytes(fid: BinaryIO, num_bytes: int, format_char_sequence: str, endian_character: str = "<") -> tuple:
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
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """Read camera parameters from a COLMAP binary file.
    
    Args:
        path: Path to the cameras.bin file
        
    Returns:
        Dictionary mapping camera_id to Camera objects
    """
    cameras = {}
    
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_cameras):
            camera_properties = _read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            # Get model name from ID
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            
            # Read parameters
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=int(width),
                height=int(height),
                params=list(params)
            )
    
    return cameras


def read_images_binary(path: str, only_3d_features:bool) -> Dict[int, Image]:
    """Read image data from a COLMAP binary file.
    
    Args:
        path: Path to the images.bin file
        
    Returns:
        Dictionary mapping image_id to Image objects
    """
    images = {}

    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_reg_images):
            binary_image_properties = _read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            
            # Read image name
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]
            
            # Read points
            num_points2D = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = _read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            
            # Parse points
            xys = []
            point3D_ids = []

            for i in range(num_points2D):
                x = x_y_id_s[3*i]
                y = x_y_id_s[3*i+1]
                point3D_id = x_y_id_s[3*i+2]

                if only_3d_features and point3D_id == INVALID_POINT3D_ID:
                    continue

                xys.append((x, y))
                point3D_ids.append(point3D_id)

            images[image_id] = Image(
                id=image_id,
                name=image_name,
                camera_id=camera_id,
                qvec=qvec,
                tvec=tvec,
                xys=xys,
                point3D_ids=point3D_ids
            )
    
    return images


def read_points3D_binary(path: str) -> Dict[int, Point3D]:
    """Read 3D points from a COLMAP binary file.
    
    Args:
        path: Path to the points3D.bin file
        
    Returns:
        Dictionary mapping point3D_id to Point3D objects
    """
    points3D = {}
    
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_points):
            binary_point_line_properties = _read_next_bytes(fid, num_bytes=43, format_char_sequence="qdddBBBd")
            
            point3D_id = binary_point_line_properties[0]
            xyz = binary_point_line_properties[1:4]
            rgb = binary_point_line_properties[4:7]
            error = binary_point_line_properties[7]
            
            # Read track
            track_length = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = _read_next_bytes(
                fid, num_bytes=8*track_length, format_char_sequence="ii"*track_length)
            
            # Parse track
            image_ids = []
            point2D_idxs = []
            
            for i in range(track_length):
                image_ids.append(track_elems[2*i])
                point2D_idxs.append(track_elems[2*i+1])
            
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs
            )
    
    return points3D


def write_cameras_binary(cameras: Dict[int, Camera], path: str) -> None:
    """Write camera parameters to a COLMAP binary file.
    
    Args:
        cameras: Dictionary of Camera objects
        path: Output file path
    """
    with open(path, "wb") as fid:
        # Write number of cameras
        fid.write(struct.pack("<Q", len(cameras)))
        
        for camera_id, camera in cameras.items():
            # Get model ID
            model_id = -1
            for camera_model in CAMERA_MODEL_NAMES.values():
                if camera_model.model_name == camera.model:
                    model_id = camera_model.model_id
                    break
            
            if model_id == -1:
                raise ValueError(f"Camera model {camera.model} not recognized")
            
            # Write camera properties
            fid.write(struct.pack("<i", camera.id))
            fid.write(struct.pack("<i", model_id))
            fid.write(struct.pack("<Q", camera.width))
            fid.write(struct.pack("<Q", camera.height))
            
            # Write camera parameters
            for param in camera.params:
                fid.write(struct.pack("<d", param))


def write_images_binary(images: Dict[int, Image], path: str) -> None:
    """Write image data to a COLMAP binary file.
    
    Args:
        images: Dictionary of Image objects
        path: Output file path
    """
    with open(path, "wb") as fid:
        # Write number of images
        fid.write(struct.pack("<Q", len(images)))
        
        for image_id, image in images.items():
            # Write image properties
            fid.write(struct.pack("<i", image.id))
            for q in image.qvec:
                fid.write(struct.pack("<d", q))
            for t in image.tvec:
                fid.write(struct.pack("<d", t))
            fid.write(struct.pack("<i", image.camera_id))
            
            # Write image name as null-terminated string
            for char in image.name:
                fid.write(struct.pack("<c", char.encode("utf-8")))
            fid.write(struct.pack("<c", b"\x00"))
            
            # Write number of points
            fid.write(struct.pack("<Q", len(image.xys)))
            
            # Write points
            for xy, point3D_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<dd", xy[0], xy[1]))
                fid.write(struct.pack("<q", point3D_id))


def write_points3D_binary(points3D: Dict[int, Point3D], path: str) -> None:
    """Write 3D points to a COLMAP binary file.
    
    Args:
        points3D: Dictionary of Point3D objects
        path: Output file path
    """
    with open(path, "wb") as fid:
        # Write number of points
        fid.write(struct.pack("<Q", len(points3D)))
        
        for point3D_id, point3D in points3D.items():
            # Write point properties
            fid.write(struct.pack("<Q", point3D.id))
            for x in point3D.xyz:
                fid.write(struct.pack("<d", x))
            for r in point3D.rgb:
                fid.write(struct.pack("<B", r))
            fid.write(struct.pack("<d", point3D.error))
            
            # Write track length
            track_length = len(point3D.image_ids)
            fid.write(struct.pack("<Q", track_length))
            
            # Write track elements
            for image_id, point2D_idx in zip(point3D.image_ids, point3D.point2D_idxs):
                fid.write(struct.pack("<ii", image_id, point2D_idx))


def read_binary_model(path: str, only_3d_features:bool) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    """Read a COLMAP binary model from a directory.
    
    Args:
        path: Directory containing the binary model files
        
    Returns:
        Tuple of (cameras, images, points3D) dictionaries
    """
    cameras_path = os.path.join(path, "cameras.bin")
    images_path = os.path.join(path, "images.bin")
    points3D_path = os.path.join(path, "points3D.bin")

    cameras = {}
    images = {}
    points3D = {}

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


def write_binary_model(cameras: Dict[int, Camera], images: Dict[int, Image], 
                      points3D: Dict[int, Point3D], path: str) -> None:
    """Write a COLMAP binary model to a directory.
    
    Args:
        cameras: Dictionary of Camera objects
        images: Dictionary of Image objects
        points3D: Dictionary of Point3D objects
        path: Output directory
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    write_cameras_binary(cameras, os.path.join(path, "cameras.bin"))
    write_images_binary(images, os.path.join(path, "images.bin"))
    write_points3D_binary(points3D, os.path.join(path, "points3D.bin"))
