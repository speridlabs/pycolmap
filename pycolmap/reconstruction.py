import numpy as np
from typing import Dict, List, Tuple, Optional 

from .image import Image
from .camera import Camera
from .point3d import Point3D
from .types import INVALID_POINT3D_ID, CAMERA_MODEL_NAMES

from .io import read_model, write_model
from .utils import detect_model_format, find_model_path, angle_between_rays


class ColmapReconstruction:
    """Main class for handling COLMAP reconstructions."""
    
    def __init__(self, path: str):
        """Initialize a COLMAP reconstruction from a directory.
        
        Args:
            path: Directory containing the COLMAP reconstruction
        """
        self.base_path = path
        self.model_path = self._find_model_path(path)

        if not self.model_path:
            raise FileNotFoundError(f"Could not find COLMAP reconstruction in {path}")
        
        self.model_format = detect_model_format(self.model_path)
        if not self.model_format:
            raise ValueError(f"Could not detect model format in {self.model_path}")
        
        self.cameras, self.images, self.points3D = read_model(self.model_path, self.model_format)
        self.name_to_image_id = {image.name: img_id for img_id, image in self.images.items()}
        self.last_camera_id = max(self.cameras.keys()) if self.cameras else 0
        self.last_image_id = max(self.images.keys()) if self.images else 0
    
    def _find_model_path(self, path: str) -> Optional[str]:
        """Find the model directory in the given path.
        
        Args:
            path: Base directory to search in
            
        Returns:
            Path to the directory containing the model, or None if not found
        """
        return find_model_path(path)
    
    def save(self, output_path: Optional[str] = None, binary: bool = True) -> None:
        """Save the current reconstruction.
        
        Args:
            output_path: Directory to save the reconstruction to (defaults to original path)
            binary: Whether to use binary format (True) or text format (False)
        """
        if output_path is None:
            output_path = self.model_path
        
        write_model(self.cameras, self.images, self.points3D, output_path, binary)
    
    def get_image_ids(self) -> List[int]:
        """Get all image IDs in the reconstruction.
        
        Returns:
            List of image IDs
        """
        return list(self.images.keys())
    
    def get_camera_ids(self) -> List[int]:
        """Get all camera IDs in the reconstruction.
        
        Returns:
            List of camera IDs
        """
        return list(self.cameras.keys())
    
    def get_point3D_ids(self) -> List[int]:
        """Get all 3D point IDs in the reconstruction.
        
        Returns:
            List of 3D point IDs
        """
        return list(self.points3D.keys())
    
    def get_image_by_name(self, name: str) -> Tuple[int, Image]:
        """Get an image by its name.
        
        Args:
            name: Image filename
            
        Returns:
            Tuple of (image_id, Image object)
            
        Raises:
            ValueError: If image with the given name is not found
        """
        image_id = self.name_to_image_id.get(name)
        if image_id is None:
            raise ValueError(f"Image with name {name} not found")
        return image_id, self.images[image_id]
    
    def get_intrinsics(self, camera_id: int) -> np.ndarray:
        """Get the intrinsic calibration matrix for a camera.
        
        Args:
            camera_id: ID of the camera
            
        Returns:
            3x3 intrinsic matrix
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Camera ID {camera_id} not found")
        
        return self.cameras[camera_id].get_calibration_matrix()
    
    def get_world_to_camera(self, image_id: int) -> np.ndarray:
        """Get the world-to-camera transformation matrix for an image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            4x4 transformation matrix
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")
        
        return self.images[image_id].get_world_to_camera_matrix()
    
    def get_camera_to_world(self, image_id: int) -> np.ndarray:
        """Get the camera-to-world transformation matrix for an image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            4x4 transformation matrix
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")
        
        return self.images[image_id].get_camera_to_world_matrix()
    
    def get_camera_center(self, image_id: int) -> Tuple[float, float, float]:
        """Get the camera center for an image in world coordinates.
        
        Args:
            image_id: ID of the image
            
        Returns:
            Camera center as (x, y, z)
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")
        
        return self.images[image_id].get_camera_center()
    
    def add_camera(self, model: str, width: int, height: int, params: List[float]) -> int:
        """Add a new camera to the reconstruction.
        
        Args:
            model: Camera model name (e.g., 'PINHOLE')
            width: Image width
            height: Image height
            params: Camera parameters
            
        Returns:
            ID of the new camera
        """
        # Validate model
        if model not in CAMERA_MODEL_NAMES:
            raise ValueError(f"Unknown camera model: {model}")
        
        # Validate parameters
        num_expected_params = CAMERA_MODEL_NAMES[model].num_params
        if len(params) != num_expected_params:
            raise ValueError(f"Camera model {model} requires {num_expected_params} parameters, got {len(params)}")
        
        # Create new camera
        self.last_camera_id += 1
        camera_id = self.last_camera_id
        
        # Add camera
        self.cameras[camera_id] = Camera(
            id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params
        )
        
        return camera_id
    
    def add_image(self, 
                 name: str, 
                 camera_id: int, 
                 qvec: Tuple[float, float, float, float], 
                 tvec: Tuple[float, float, float], 
                 xys: Optional[List[Tuple[float, float]]] = None, 
                 point3D_ids: Optional[List[int]] = None) -> int:
        """Add a new image to the reconstruction.
        
        Args:
            name: Image filename
            camera_id: ID of the camera used
            qvec: Quaternion for rotation [w, x, y, z]
            tvec: Translation vector [x, y, z]
            xys: 2D point coordinates
            point3D_ids: IDs of corresponding 3D points
            
        Returns:
            ID of the new image
        """
        # Validate camera
        if camera_id not in self.cameras:
            raise ValueError(f"Camera ID {camera_id} not found")
        
        # Default empty lists
        xys = xys if xys is not None else []
        point3D_ids = point3D_ids if point3D_ids is not None else []
        
        # Validate lengths
        if len(xys) != len(point3D_ids):
            raise ValueError(f"Number of 2D points ({len(xys)}) does not match number of 3D point IDs ({len(point3D_ids)})")
        
        # Create new image
        self.last_image_id += 1
        image_id = self.last_image_id
        
        # Add image
        self.images[image_id] = Image(
            id=image_id,
            name=name,
            camera_id=camera_id,
            qvec=qvec,
            tvec=tvec,
            xys=xys,
            point3D_ids=point3D_ids
        )
        
        # Update lookup
        self.name_to_image_id[name] = image_id
        
        return image_id
    
    def delete_image(self, image_id: int) -> None:
        """Delete an image from the reconstruction.
        
        Args:
            image_id: ID of the image to delete
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")
        
        # Get image
        image = self.images[image_id]
        
        # Update 3D points
        for point3D_id in set(image.point3D_ids):
            if point3D_id == INVALID_POINT3D_ID:
                continue
                
            if point3D_id in self.points3D:
                point3D = self.points3D[point3D_id]
                
                # Find indices of this image in the track
                indices_to_remove = [i for i, img_id in enumerate(point3D.image_ids) if img_id == image_id]
                
                if indices_to_remove:
                    # Remove this image from the track
                    new_image_ids = [img_id for i, img_id in enumerate(point3D.image_ids) if i not in indices_to_remove]
                    new_point2D_idxs = [idx for i, idx in enumerate(point3D.point2D_idxs) if i not in indices_to_remove]
                    
                    # If track becomes empty, delete the point
                    if not new_image_ids:
                        del self.points3D[point3D_id]
                    else:
                        # Update the point
                        self.points3D[point3D_id] = Point3D(
                            id=point3D.id,
                            xyz=point3D.xyz,
                            rgb=point3D.rgb,
                            error=point3D.error,
                            image_ids=new_image_ids,
                            point2D_idxs=new_point2D_idxs
                        )
        
        # Remove image from name lookup
        if image.name in self.name_to_image_id:
            del self.name_to_image_id[image.name]
        
        # Delete image
        del self.images[image_id]
    
    def delete_camera(self, camera_id: int, delete_images: bool = False) -> None:
        """Delete a camera from the reconstruction.
        
        Args:
            camera_id: ID of the camera to delete
            delete_images: Whether to also delete images using this camera
        """
        if camera_id not in self.cameras:
            raise ValueError(f"Camera ID {camera_id} not found")
        
        # Check if camera is in use
        images_using_camera = [img_id for img_id, img in self.images.items() if img.camera_id == camera_id]
        
        if images_using_camera:
            if delete_images:
                # Delete all images using this camera
                for image_id in images_using_camera:
                    self.delete_image(image_id)
            else:
                raise ValueError(f"Cannot delete camera {camera_id} because it is used by {len(images_using_camera)} images")
        
        # Delete camera
        del self.cameras[camera_id]
    
    def delete_point3D(self, point3D_id: int) -> None:
        """Delete a 3D point from the reconstruction.
        
        Args:
            point3D_id: ID of the 3D point to delete
        """
        if point3D_id not in self.points3D:
            raise ValueError(f"Point3D ID {point3D_id} not found")
        
        # Get 3D point
        point3D = self.points3D[point3D_id]
        
        # Update images
        for image_id, point2D_idx in zip(point3D.image_ids, point3D.point2D_idxs):
            if image_id in self.images:
                image = self.images[image_id]
                
                # Find 2D point corresponding to this 3D point
                indices = [i for i, p3d_id in enumerate(image.point3D_ids) if p3d_id == point3D_id]
                
                for idx in indices:
                    # Mark 2D point as not having a 3D point
                    point3D_ids = list(image.point3D_ids)
                    point3D_ids[idx] = INVALID_POINT3D_ID
                    
                    # Update image
                    self.images[image_id] = Image(
                        id=image.id,
                        name=image.name,
                        camera_id=image.camera_id,
                        qvec=image.qvec,
                        tvec=image.tvec,
                        xys=image.xys,
                        point3D_ids=point3D_ids
                    )
        
        # Delete 3D point
        del self.points3D[point3D_id]
    
    def filter_points(self, 
                     min_track_length: int = 3, 
                     max_error: float = float('inf'),
                     min_angle: float = 0.0) -> int:
        """Filter 3D points based on quality criteria.
        
        Args:
            min_track_length: Minimum number of images that must observe a point
            max_error: Maximum reprojection error
            min_angle: Minimum triangulation angle in degrees
            
        Returns:
            Number of points removed
        """
        points_to_remove = []
        
        for point3D_id, point3D in self.points3D.items():
            # Check track length
            if point3D.get_track_length() < min_track_length:
                points_to_remove.append(point3D_id)
                continue
            
            # Check reprojection error
            if point3D.error > max_error:
                points_to_remove.append(point3D_id)
                continue
            
            # Check triangulation angle
            if min_angle > 0 and point3D.get_track_length() >= 2:
                max_angle_found = 0.0
                centers = []
                
                # Get camera centers
                for image_id in point3D.image_ids:
                    if image_id in self.images:
                        centers.append(self.get_camera_center(image_id))
                
                # Check all pairs of cameras
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        # Calculate angle
                        angle = angle_between_rays(centers[i], centers[j], point3D.xyz)
                        max_angle_found = max(max_angle_found, angle)
                
                # If max angle is below threshold, remove point
                if max_angle_found < min_angle:
                    points_to_remove.append(point3D_id)
        
        # Delete filtered points
        for point3D_id in points_to_remove:
            self.delete_point3D(point3D_id)
        
        return len(points_to_remove)
    
    def get_points3D_for_image(self, image_id: int) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """Get 3D points visible in an image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            Tuple of (points, colors) where points is a list of 3D coordinates
            and colors is a list of RGB values
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")
        
        image = self.images[image_id]
        points = []
        colors = []
        
        for point3D_id in image.point3D_ids:
            if point3D_id != INVALID_POINT3D_ID and point3D_id in self.points3D:
                point3D = self.points3D[point3D_id]
                points.append(point3D.xyz)
                colors.append(point3D.rgb)
        
        return points, colors
    
    def get_observations_for_point3D(self, point3D_id: int) -> List[Tuple[int, int, Tuple[float, float]]]:
        """Get all observations of a 3D point.
        
        Args:
            point3D_id: ID of the 3D point
            
        Returns:
            List of tuples (image_id, point2D_idx, xy) for each observation
        """
        if point3D_id not in self.points3D:
            raise ValueError(f"Point3D ID {point3D_id} not found")
        
        point3D = self.points3D[point3D_id]
        observations = []
        
        for image_id, point2D_idx in zip(point3D.image_ids, point3D.point2D_idxs):
            if image_id in self.images:
                image = self.images[image_id]
                if 0 <= point2D_idx < len(image.xys):
                    observations.append((image_id, point2D_idx, image.xys[point2D_idx]))
        
        return observations
    
    def get_reprojection_errors(self, image_id: int) -> List[float]:
        """Get reprojection errors for all 3D points visible in an image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            List of reprojection errors
        """
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")
        
        image = self.images[image_id]
        camera = self.cameras[image.camera_id]
        P = np.dot(camera.get_calibration_matrix(), image.get_rotation_matrix())
        errors = []
        
        for xy, point3D_id in zip(image.xys, image.point3D_ids):
            if point3D_id != INVALID_POINT3D_ID and point3D_id in self.points3D:
                point3D = self.points3D[point3D_id]
                
                # Project 3D point to image
                X = np.array([*point3D.xyz, 1.0])
                x = np.dot(P, X)
                x = x[:2] / x[2]
                
                # Calculate reprojection error
                error = np.sqrt((x[0] - xy[0])**2 + (x[1] - xy[1])**2)
                errors.append(error)
        
        return errors
    
    def calculate_point_cloud_stats(self) -> Dict[str, float]:
        """Calculate statistics for the point cloud.
        
        Returns:
            Dictionary with statistics (num_points, mean_track_length, mean_error)
        """
        if not self.points3D:
            return {
                "num_points": 0,
                "mean_track_length": 0.0,
                "mean_error": 0.0
            }
        
        num_points = len(self.points3D)
        track_lengths = [point3D.get_track_length() for point3D in self.points3D.values()]
        errors = [point3D.error for point3D in self.points3D.values()]
        
        return {
            "num_points": num_points,
            "mean_track_length": sum(track_lengths) / num_points,
            "mean_error": sum(errors) / num_points
        }
    
    def __str__(self) -> str:
        """Get a string representation of the reconstruction.
        
        Returns:
            String summary of the reconstruction
        """
        return (f"COLMAP Reconstruction:\n"
                f"  Model path: {self.model_path}\n"
                f"  Model format: {self.model_format}\n"
                f"  Cameras: {len(self.cameras)}\n"
                f"  Images: {len(self.images)}\n"
                f"  3D Points: {len(self.points3D)}")
