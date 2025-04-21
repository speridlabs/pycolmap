import os
import tempfile
import shutil
import struct

from pycolmap import Camera, Image, Point3D


def create_mock_binary_reconstruction(output_dir):
    """
    Create a minimal binary COLMAP reconstruction for testing.
    
    Args:
        output_dir: Directory where the mock data will be saved
    
    Returns:
        Tuple of (cameras, images, points3D) dictionaries
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a camera
    cameras = {
        1: Camera(
            id=1,
            model="PINHOLE",
            width=1920,
            height=1080,
            params=[1000.0, 1000.0, 960.0, 540.0]
        )
    }
    
    # Create images
    images = {
        1: Image(
            id=1,
            name="image1.jpg",
            camera_id=1,
            qvec=(1.0, 0.0, 0.0, 0.0),
            tvec=(0.0, 0.0, 0.0),
            xys=[(100.0, 200.0), (300.0, 400.0)],
            point3D_ids=[1, -1]
        ),
        2: Image(
            id=2,
            name="image2.jpg",
            camera_id=1,
            qvec=(0.9, 0.1, 0.0, 0.0),
            tvec=(1.0, 0.0, 0.0),
            xys=[(150.0, 250.0)],
            point3D_ids=[1]
        )
    }
    
    # Create 3D points
    points3D = {
        1: Point3D(
            id=1,
            xyz=(1.0, 2.0, 3.0),
            rgb=(255, 0, 0),
            error=0.5,
            image_ids=[1, 2],
            point2D_idxs=[0, 0]
        )
    }

    # Write binary files
    write_cameras_binary(cameras, os.path.join(output_dir, "cameras.bin"))
    write_images_binary(images, os.path.join(output_dir, "images.bin"))
    write_points3D_binary(points3D, os.path.join(output_dir, "points3D.bin"))
    
    return cameras, images, points3D


def write_cameras_binary(cameras, path):
    """Write cameras to a binary file."""
    with open(path, "wb") as f:
        # Write number of cameras
        f.write(struct.pack("<Q", len(cameras)))
        
        for camera_id, camera in cameras.items():
            # Get model ID
            model_id_map = {
                "SIMPLE_PINHOLE": 0,
                "PINHOLE": 1,
                "SIMPLE_RADIAL": 2,
                "RADIAL": 3,
                "OPENCV": 4,
                "OPENCV_FISHEYE": 5
            }
            model_id = model_id_map.get(camera.model, 0)
            
            # Write camera data
            f.write(struct.pack("<I", camera.id))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", camera.width))
            f.write(struct.pack("<Q", camera.height))
            
            # Write parameters
            for param in camera.params:
                f.write(struct.pack("<d", param))


def write_images_binary(images, path):
    """Write images to a binary file."""
    with open(path, "wb") as f:
        # Write number of images
        f.write(struct.pack("<Q", len(images)))
        
        for image_id, image in images.items():
            # Write image properties
            f.write(struct.pack("<I", image.id))
            for q in image.qvec:
                f.write(struct.pack("<d", q))
            for t in image.tvec:
                f.write(struct.pack("<d", t))
            f.write(struct.pack("<I", image.camera_id))
            
            # Write image name
            for c in image.name:
                f.write(struct.pack("<c", c.encode("utf-8")))
            f.write(struct.pack("<c", b"\0"))
            
            # Write keypoints
            f.write(struct.pack("<Q", len(image.xys)))
            for xy, point3D_id in zip(image.xys, image.point3D_ids):
                f.write(struct.pack("<dd", xy[0], xy[1]))
                f.write(struct.pack("<q", point3D_id))


def write_points3D_binary(points3D, path):
    """Write 3D points to a binary file."""
    with open(path, "wb") as f:
        # Write number of points
        f.write(struct.pack("<Q", len(points3D)))
        
        for point3D_id, point3D in points3D.items():
            # Write point data
            f.write(struct.pack("<Q", point3D.id))
            for x in point3D.xyz:
                f.write(struct.pack("<d", x))
            for c in point3D.rgb:
                f.write(struct.pack("<B", c))
            f.write(struct.pack("<d", point3D.error))
            
            # Write track data
            track_length = len(point3D.image_ids)
            f.write(struct.pack("<Q", track_length))
            for img_id, point2D_idx in zip(point3D.image_ids, point3D.point2D_idxs):
                f.write(struct.pack("<II", img_id, point2D_idx))


class MockDataTest:
    """Class to handle creation and cleanup of mock data for testing."""
    
    def __init__(self):
        self.temp_dir = None
        self.cameras = None
        self.images = None
        self.points3D = None
    
    def setup(self):
        """Set up mock data."""
        self.temp_dir = tempfile.mkdtemp()
        self.cameras, self.images, self.points3D = create_mock_binary_reconstruction(self.temp_dir)
        return self.temp_dir
    
    def cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
