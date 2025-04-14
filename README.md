# pycolmap

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for working with COLMAP reconstructions, providing a clean and intuitive API for loading, manipulating, and saving COLMAP projects in both binary and text formats.

## Features

- **Format Support**: Load and save COLMAP reconstructions in both binary and text formats
- **Automatic Detection**: Intelligently finds model files in standard locations (`sparse/0`, `sparse`, or root directory)
- **Complete Data Access**: Easy access to cameras, images, and 3D points with type-safe classes
- **Manipulation API**: Add/delete cameras, images, and 3D points with proper reference updating
- **Filtering**: Remove low-quality 3D points based on track length, reprojection error, and triangulation angle
- **Coordinate Transformations**: Convert between world and camera coordinate systems
- **Consistency Verification**: Optional validation to ensure data integrity
- **Comprehensive Type Annotations**: Full typing support for better IDE integration

## Installation

### From PyPI

```bash
pip install https://github.com/speridlabs/pycolmap.git
```

### From Source

```bash
git clone https://github.com/speridlabs/pycolmap.git
cd pycolmap
pip install -e .
```

## Quick Start

```python
import pycolmap

# Load a COLMAP reconstruction (automatically detects binary/text format)
reconstruction = pycolmap.ColmapReconstruction("/path/to/colmap/project")

# Access cameras, images, and points
for camera_id, camera in reconstruction.cameras.items():
    print(f"Camera {camera_id}: {camera.model} {camera.width}x{camera.height}")

for image_id, image in reconstruction.images.items():
    print(f"Image {image_id}: {image.name}, {len(image.xys)} features")

# Get statistics
stats = reconstruction.get_statistics()
print(f"Number of points: {stats['num_points3D']}")
print(f"Mean reprojection error: {stats['mean_reprojection_error']:.2f} pixels")

# Filter points with poor quality
reconstruction.filter_points3D(
    min_track_len=3,    # Must be seen in at least 3 images
    max_error=5.0,      # Maximum reprojection error of 5 pixels
    min_angle=3.0       # Minimum triangulation angle of 3 degrees
)

# Save the modified reconstruction
reconstruction.save("/path/to/output", binary=True)
```

## Basic Usage Examples

### Loading a Model

```python
# Auto-detect location and format
reconstruction = pycolmap.ColmapReconstruction("/path/to/colmap/project")

# Access components by ID
camera = reconstruction.cameras[1]
image = reconstruction.images[1]
point3D = reconstruction.points3D[1]

# Find image by name
image = reconstruction.get_image_by_name("image.jpg")
```

### Adding Components

```python
# Add a new camera
camera_id = reconstruction.add_camera(
    model="PINHOLE",
    width=1920, 
    height=1080, 
    params=[1000.0, 1000.0, 960.0, 540.0]  # fx, fy, cx, cy
)

# Add a new image
image_id = reconstruction.add_image(
    name="new_image.jpg",
    camera_id=camera_id,
    qvec=(1.0, 0.0, 0.0, 0.0),  # Quaternion [w, x, y, z]
    tvec=(0.0, 0.0, 0.0)        # Translation vector
)

# Add a new 3D point
point3D_id = reconstruction.add_point3D(
    xyz=(1.0, 2.0, 3.0),
    track=[(image_id, 0), (image_id, 1)],  # List of (image_id, point2D_idx)
    rgb=(255, 0, 0)  # Red point
)
```

### Working with Coordinates

```python
# Get camera matrix (K)
K = camera.get_calibration_matrix()

# Get world-to-camera transform
W2C = image.get_world_to_camera_matrix()

# Get camera-to-world transform
C2W = image.get_camera_to_world_matrix()

# Get camera center in world coordinates
center = image.get_camera_center()
```

### Manipulating the Reconstruction

```python
# Delete components
reconstruction.delete_camera(camera_id)
reconstruction.delete_image(image_id)
reconstruction.delete_point3D(point3D_id)

# Filter 3D points
num_removed = reconstruction.filter_points3D(
    min_track_len=3,
    max_error=2.0,
    min_angle=5.0
)
print(f"Removed {num_removed} low-quality points")
```

### Saving Changes

```python
# Save in the original location with the original format
reconstruction.save()

# Save in a new location with binary format
reconstruction.save("/path/to/output", binary=True)

# Save in a new location with text format
reconstruction.save("/path/to/output", binary=False)
```

## Advanced Usage

### Direct I/O Functions

For lower-level access, the library provides direct I/O functions:

```python
from pycolmap.io import read_model, write_model

# Read a model in any format
cameras, images, points3D = read_model("/path/to/model")

# Read a model in a specific format
cameras, images, points3D = read_model("/path/to/model", file_format=".bin")

# Write a model
write_model(cameras, images, points3D, "/path/to/output", binary=True)
```

### Working with Point Tracks

```python
# Get all observations of a 3D point
observations = reconstruction.get_observations_for_point3D(point3D_id)
for image_id, point2D_idx, xy in observations:
    print(f"Point {point3D_id} observed in image {image_id} at {xy}")

# Get track length
track_length = point3D.get_track_length()

# Get valid 3D points in an image
valid_points = image.get_valid_points3D()
for point3D_id, xy in valid_points:
    print(f"Point {point3D_id} at image coordinates {xy}")
```

### Utility Functions

```python
from pycolmap.utils import qvec2rotmat, rotmat2qvec, angle_between_rays, triangulate_point

# Convert quaternion to rotation matrix
R = qvec2rotmat((1.0, 0.0, 0.0, 0.0))

# Convert rotation matrix to quaternion
qvec = rotmat2qvec(R)

# Calculate angle between camera rays
angle = angle_between_rays(
    camera1_center=(0, 0, 0),
    camera2_center=(1, 0, 0),
    point3D=(1, 1, 5)
)

# Triangulate a 3D point from two 2D observations
P1 = get_projection_matrix(camera1, image1)
P2 = get_projection_matrix(camera2, image2)
point3D = triangulate_point(P1, P2, point1_xy, point2_xy)
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
