import os
from typing import Dict, Tuple 

from ..image import Image
from ..camera import Camera
from ..point3d import Point3D


def read_cameras_text(path: str) -> Dict[int, Camera]:
    """Read camera parameters from a COLMAP text file.
    
    Args:
        path: Path to the cameras.txt file
        
    Returns:
        Dictionary mapping camera_id to Camera objects
    """
    cameras = {}
    
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Parse camera data
            try:
                elems = line.split()
                # Basic check for minimum expected elements
                if len(elems) < 4: 
                    # Log or warn about skipped line if desired
                    continue 
                    
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                # Ensure params are floats
                params = [float(p) for p in elems[4:]] 
                
                # Create camera object
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params
                )
            except (ValueError, IndexError) as e:
                # Log or warn about the error and the skipped line if desired
                # print(f"Warning: Skipping malformed camera line: {line} - Error: {e}")
                continue # Skip to the next line
    
    return cameras


def read_images_text(path: str) -> Dict[int, Image]:
    """Read image data from a COLMAP text file.
    
    Args:
        path: Path to the images.txt file
        
    Returns:
        Dictionary mapping image_id to Image objects
    """
    images = {}
    
    with open(path, "r") as fid:
        read_points = False
        # Initialize variables to ensure they are always bound
        image_id: int = -1 
        qvec: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        tvec: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        camera_id: int = -1
        image_name: str = ""
        
        for line in fid:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            if read_points:
                # Read points (second line of image data)
                elems = line.split()
                
                if len(elems) == 0:
                    # No points
                    xys = []
                    point3D_ids = []
                else:
                    # Parse points
                    xys = []
                    point3D_ids = []
                    
                    # Ensure elems has a length that is a multiple of 3
                    if len(elems) % 3 == 0:
                        for i in range(0, len(elems), 3):
                            try:
                                x = float(elems[i])
                                y = float(elems[i+1])
                                point3D_id = int(elems[i+2])
                                
                                xys.append((x, y))
                                # Use -1 for invalid point3D_id if necessary, as per COLMAP format
                                point3D_ids.append(point3D_id if point3D_id != -1 else -1) 
                            except (ValueError, IndexError):
                                # Skip malformed point data triplets
                                continue
                    # else: Handle or log malformed points line if needed

                # Create image object only if image_id was properly set
                # (i.e., the first line was read successfully)
                if image_id != -1:
                    images[image_id] = Image(
                    id=image_id,
                    name=image_name,
                    camera_id=camera_id,
                    qvec=qvec,
                    tvec=tvec,
                    xys=xys,
                    point3D_ids=point3D_ids
                )
                
                read_points = False
                
            else:
                # Read image properties (first line of image data)
                elems = line.split()
                
                # Reset image_id for the new entry attempt
                image_id = -1 
                
                # Check if there are enough elements for basic properties
                if len(elems) >= 10:
                    try:
                        image_id = int(elems[0])
                        qvec = (float(elems[1]), float(elems[2]), float(elems[3]), float(elems[4]))
                        tvec = (float(elems[5]), float(elems[6]), float(elems[7]))
                        camera_id = int(elems[8])
                        image_name = elems[9]
                        
                        # Successfully parsed the first line, expect points next
                        read_points = True 
                    except ValueError:
                        # Handle potential float/int conversion errors for this line
                        # Reset read_points and skip to the next line
                        read_points = False 
                        continue
                else:
                    # Line doesn't have enough elements, skip it
                    read_points = False
                    continue # Skip to the next line
    
    return images


def read_points3D_text(path: str) -> Dict[int, Point3D]:
    """Read 3D points from a COLMAP text file.
    
    Args:
        path: Path to the points3D.txt file
        
    Returns:
        Dictionary mapping point3D_id to Point3D objects
    """
    points3D = {}
    
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Parse 3D point data
            try:
                elems = line.split()
                # Basic check for minimum expected elements (ID, XYZ, RGB, ERROR)
                if len(elems) < 8:
                    # Log or warn about skipped line if desired
                    continue

                point3D_id = int(elems[0])
                xyz = (float(elems[1]), float(elems[2]), float(elems[3]))
                rgb = (int(elems[4]), int(elems[5]), int(elems[6]))
                error = float(elems[7])
                
                # Parse track - ensure track data has pairs
                image_ids = []
                point2D_idxs = []
                if (len(elems) - 8) % 2 != 0:
                    # Log or warn about malformed track data if desired
                    # Skip track parsing for this point or handle as appropriate
                    pass # Or raise an error, or skip the point entirely
                else:
                    for i in range(8, len(elems), 2):
                        image_ids.append(int(elems[i]))
                        point2D_idxs.append(int(elems[i+1]))
                
                # Create 3D point object
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs
                )
            except (ValueError, IndexError) as e:
                # Log or warn about the error and the skipped line if desired
                # print(f"Warning: Skipping malformed points3D line: {line} - Error: {e}")
                continue # Skip to the next line
    
    return points3D


def write_cameras_text(cameras: Dict[int, Camera], path: str) -> None:
    """Write camera parameters to a COLMAP text file.
    
    Args:
        cameras: Dictionary of Camera objects
        path: Output file path
    """
    with open(path, "w") as fid:
        # Write header
        fid.write("# Camera list with one line of data per camera:\n")
        fid.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fid.write(f"# Number of cameras: {len(cameras)}\n")
        
        # Write cameras
        for camera_id, camera in sorted(cameras.items()):
            params_str = " ".join(map(str, camera.params))
            fid.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params_str}\n")


def write_images_text(images: Dict[int, Image], path: str) -> None:
    """Write image data to a COLMAP text file.
    
    Args:
        images: Dictionary of Image objects
        path: Output file path
    """
    with open(path, "w") as fid:
        # Write header
        fid.write("# Image list with two lines of data per image:\n")
        fid.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fid.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fid.write(f"# Number of images: {len(images)}\n")
        
        # Write images
        for image_id, image in sorted(images.items()):
            # Write image properties
            qvec_str = " ".join(map(str, image.qvec))
            tvec_str = " ".join(map(str, image.tvec))
            fid.write(f"{image.id} {qvec_str} {tvec_str} {image.camera_id} {image.name}\n")
            
            # Write points
            points_str = ""
            for xy, point3D_id in zip(image.xys, image.point3D_ids):
                points_str += f"{xy[0]} {xy[1]} {point3D_id} "
            
            # Write points (or empty line for no points)
            fid.write(f"{points_str}\n")


def write_points3D_text(points3D: Dict[int, Point3D], path: str) -> None:
    """Write 3D points to a COLMAP text file.
    
    Args:
        points3D: Dictionary of Point3D objects
        path: Output file path
    """
    with open(path, "w") as fid:
        # Write header
        fid.write("# 3D point list with one line of data per point:\n")
        fid.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        fid.write(f"# Number of points: {len(points3D)}\n")
        
        # Write points
        for point3D_id, point3D in sorted(points3D.items()):
            # Write point properties
            xyz_str = " ".join(map(str, point3D.xyz))
            rgb_str = " ".join(map(str, point3D.rgb))
            
            # Write track
            track_str = ""
            for image_id, point2D_idx in zip(point3D.image_ids, point3D.point2D_idxs):
                track_str += f"{image_id} {point2D_idx} "
            
            # Write point
            fid.write(f"{point3D.id} {xyz_str} {rgb_str} {point3D.error} {track_str}\n")


def read_text_model(path: str) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    """Read a COLMAP text model from a directory.
    
    Args:
        path: Directory containing the text model files
        
    Returns:
        Tuple of (cameras, images, points3D) dictionaries
    """
    cameras = read_cameras_text(os.path.join(path, "cameras.txt"))
    images = read_images_text(os.path.join(path, "images.txt"))
    points3D = read_points3D_text(os.path.join(path, "points3D.txt"))
    
    return cameras, images, points3D


def write_text_model(cameras: Dict[int, Camera], images: Dict[int, Image], 
                    points3D: Dict[int, Point3D], path: str) -> None:
    """Write a COLMAP text model to a directory.
    
    Args:
        cameras: Dictionary of Camera objects
        images: Dictionary of Image objects
        points3D: Dictionary of Point3D objects
        path: Output directory
    """
    if not os.path.exists(path):
        os.makedirs(path)
    
    write_cameras_text(cameras, os.path.join(path, "cameras.txt"))
    write_images_text(images, os.path.join(path, "images.txt"))
    write_points3D_text(points3D, os.path.join(path, "points3D.txt"))
