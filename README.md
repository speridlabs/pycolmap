# COLMAP Utils

A Python library for working with COLMAP reconstructions, providing a clean and intuitive API for loading, manipulating, and saving COLMAP projects.

## Features

- Load COLMAP reconstructions from binary or text formats
- Access and manipulate cameras, images, and 3D points
- Add or delete cameras and images
- Filter 3D points based on quality metrics
- Convert between coordinate systems
- Save reconstructions in binary or text formats

## Installation

```bash
# From PyPI (when published)
pip install pycolmap

# From source
git clone https://github.com/speridlabs/pycolmap.git
cd pycolmap
pip install -e .
```

## Usage Examples

### Loading a COLMAP Reconstruction

```python
from colmap_utils import ColmapReconstruction

# Load a COLMAP reconstruction (automatically detects binary/text format)
reconstruction = ColmapReconstruction("/path/to/colmap/project")

# Access cameras, images, and points
for camera_id, camera in reconstruction.cameras.items():
    print(f"Camera {camera_id}: {camera.model} {camera.width}x{camera.height}")

for image_id, image in reconstruction.images.items():
    print(f"Image {image_id}: {image.name}")

print(f"Number of 3D points: {len(reconstruction.points3D)}")
```

### Getting Camera Matrices

```python
# Get calibration matrix for a camera
K = reconstruction.get_intrinsics(camera_id)

# Get world-to-camera transform for an image
world_to_cam = reconstruction.get_world_to_camera(image_id)

# Get camera-to-world transform for an image
cam_to_world = reconstruction.get_camera_to_world(image_id)
```

### Manipulating the Reconstruction

```python
# Add a new camera
camera_id = reconstruction.add_camera(
    model="PINHOLE",
    width=1920, 
    height=1080, 
    params=[1000, 1000, 960, 540]  # fx, fy, cx, cy
)

# Add a new image
image_id = reconstruction.add_image(
    name="new_image.jpg",
    camera_id=camera_id,
    qvec=(1.0, 0.0, 0.0, 0.0),  # Quaternion [w, x, y, z]
    tvec=(0.0, 0.0, 0.0)        # Translation vector
)

# Delete an image
reconstruction.delete_image(image_id)

# Filter 3D points
reconstruction.filter_points(
    min_track_length=3,  # Must be seen in at least 3 images
    max_error=5.0,       # Maximum reprojection error of 5 pixels
    min_angle=3.0        # Minimum triangulation angle in degrees
)

# Save the modified reconstruction
reconstruction.save("/path/to/output", binary=True)
```

### Working with 3D Points

```python
# Get 3D points visible in an image
points, colors = reconstruction.get_points3D_for_image(image_id)

# Get all 3D points
all_points = [point.xyz for point in reconstruction.points3D.values()]
```

## License

MIT License
