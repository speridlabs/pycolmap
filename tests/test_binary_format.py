import os
import unittest
import tempfile
import shutil

import pycolmap
from .mock_data import MockDataTest


class TestBinaryFormat(unittest.TestCase):
    """Tests specifically for binary format handling."""

    def setUp(self):
        """Set up test data."""
        self.mock_data = MockDataTest()
        self.test_dir = self.mock_data.setup()
    
    def tearDown(self):
        """Clean up test data."""
        self.mock_data.cleanup()
    
    def test_direct_load_binary(self):
        """Test loading binary files directly."""
        # Use direct IO functions
        cameras = pycolmap.io.read_cameras_binary(os.path.join(self.test_dir, "cameras.bin"))
        images = pycolmap.io.read_images_binary(os.path.join(self.test_dir, "images.bin"))
        points3D = pycolmap.io.read_points3D_binary(os.path.join(self.test_dir, "points3D.bin"))
        
        # Check counts
        self.assertEqual(len(cameras), 1)
        self.assertEqual(len(images), 2)
        self.assertEqual(len(points3D), 1)
        
        # Check some values
        self.assertEqual(cameras[1].params[0], 1000.0)  # fx
        self.assertEqual(images[1].name, "image1.jpg")
        self.assertEqual(points3D[1].xyz[2], 3.0)  # z coordinate
    
    def test_format_detection(self):
        """Test automatic format detection."""
        # Check that format is correctly detected
        format_ext = pycolmap.utils.detect_model_format(self.test_dir)
        self.assertEqual(format_ext, ".bin")
        
        # Check that path is correctly found
        found_path = pycolmap.utils.find_model_path(self.test_dir)
        self.assertEqual(found_path, self.test_dir)
        
        # Create a nested structure
        nested_dir = os.path.join(tempfile.mkdtemp(), "sparse", "0")
        os.makedirs(nested_dir, exist_ok=True)
        
        # Copy files to the nested structure
        for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
            shutil.copy(
                os.path.join(self.test_dir, filename),
                os.path.join(nested_dir, filename)
            )
        
        # Check that nested path is correctly found
        parent_dir = os.path.dirname(os.path.dirname(nested_dir))
        found_path = pycolmap.utils.find_model_path(parent_dir)
        self.assertEqual(found_path, nested_dir)
        
        # Clean up
        shutil.rmtree(os.path.dirname(os.path.dirname(nested_dir)))
    
    def test_binary_round_trip(self):
        """Test full round-trip write and read."""
        # Load initial data
        reconstruction = pycolmap.ColmapReconstruction(self.test_dir)
        
        # Create a new directory for round-trip test
        round_trip_dir = os.path.join(tempfile.mkdtemp(), "round_trip")
        os.makedirs(round_trip_dir, exist_ok=True)
        
        # Save to new directory
        reconstruction.save(round_trip_dir, binary=True)
        
        # Load back from new directory
        reconstruction2 = pycolmap.ColmapReconstruction(round_trip_dir)
        
        # Compare
        self.assertEqual(len(reconstruction.cameras), len(reconstruction2.cameras))
        self.assertEqual(len(reconstruction.images), len(reconstruction2.images))
        self.assertEqual(len(reconstruction.points3D), len(reconstruction2.points3D))
        
        # Check specific values
        camera1 = reconstruction.cameras[1]
        camera2 = reconstruction2.cameras[1]
        self.assertEqual(camera1.model, camera2.model)
        self.assertEqual(camera1.width, camera2.width)
        self.assertEqual(camera1.height, camera2.height)
        
        # Clean up
        shutil.rmtree(os.path.dirname(round_trip_dir))


if __name__ == "__main__":
    unittest.main()
