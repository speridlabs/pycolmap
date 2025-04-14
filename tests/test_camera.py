import unittest
import numpy as np

from pycolmap import Camera


class TestCamera(unittest.TestCase):
    """Tests for Camera class."""

    def test_camera_initialization(self):
        """Test camera initialization with different models."""
        # Test PINHOLE model
        camera = Camera(
            id=1,
            model="PINHOLE",
            width=1920,
            height=1080,
            params=[1000.0, 1000.0, 960.0, 540.0]  # fx, fy, cx, cy
        )
        
        self.assertEqual(camera.id, 1)
        self.assertEqual(camera.model, "PINHOLE")
        self.assertEqual(camera.width, 1920)
        self.assertEqual(camera.height, 1080)
        self.assertEqual(len(camera.params), 4)
        
        # Test SIMPLE_PINHOLE model
        camera = Camera(
            id=2,
            model="SIMPLE_PINHOLE",
            width=1920,
            height=1080,
            params=[1000.0, 960.0, 540.0]  # f, cx, cy
        )
        
        self.assertEqual(camera.id, 2)
        self.assertEqual(camera.model, "SIMPLE_PINHOLE")
        self.assertEqual(camera.width, 1920)
        self.assertEqual(camera.height, 1080)
        self.assertEqual(len(camera.params), 3)
        
        # Test OPENCV model
        camera = Camera(
            id=3,
            model="OPENCV",
            width=1920,
            height=1080,
            params=[1000.0, 1000.0, 960.0, 540.0, 0.1, 0.01, 0.001, 0.0001]
        )
        
        self.assertEqual(camera.id, 3)
        self.assertEqual(camera.model, "OPENCV")
        self.assertEqual(camera.width, 1920)
        self.assertEqual(camera.height, 1080)
        self.assertEqual(len(camera.params), 8)
    
    def test_invalid_camera(self):
        """Test initialization with invalid parameters."""
        # Invalid model
        with self.assertRaises(ValueError):
            Camera(
                id=1,
                model="INVALID_MODEL",
                width=1920,
                height=1080,
                params=[1000.0, 1000.0, 960.0, 540.0]
            )
        
        # Wrong number of parameters
        with self.assertRaises(ValueError):
            Camera(
                id=1,
                model="PINHOLE",
                width=1920,
                height=1080,
                params=[1000.0, 1000.0, 960.0]  # Missing one parameter
            )
        
        # Invalid dimensions
        with self.assertRaises(ValueError):
            Camera(
                id=1,
                model="PINHOLE",
                width=-1,  # Negative width
                height=1080,
                params=[1000.0, 1000.0, 960.0, 540.0]
            )
    
    def test_calibration_matrix(self):
        """Test getting calibration matrix."""
        # PINHOLE model
        camera = Camera(
            id=1,
            model="PINHOLE",
            width=1920,
            height=1080,
            params=[1000.0, 1000.0, 960.0, 540.0]  # fx, fy, cx, cy
        )
        
        K = camera.get_calibration_matrix()
        
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[0, 0], 1000.0)  # fx
        self.assertEqual(K[1, 1], 1000.0)  # fy
        self.assertEqual(K[0, 2], 960.0)   # cx
        self.assertEqual(K[1, 2], 540.0)   # cy
        
        # SIMPLE_PINHOLE model
        camera = Camera(
            id=2,
            model="SIMPLE_PINHOLE",
            width=1920,
            height=1080,
            params=[1000.0, 960.0, 540.0]  # f, cx, cy
        )
        
        K = camera.get_calibration_matrix()
        
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[0, 0], 1000.0)  # f
        self.assertEqual(K[1, 1], 1000.0)  # f
        self.assertEqual(K[0, 2], 960.0)   # cx
        self.assertEqual(K[1, 2], 540.0)   # cy
    
    def test_distortion_params(self):
        """Test getting distortion parameters."""
        # PINHOLE model (no distortion)
        camera = Camera(
            id=1,
            model="PINHOLE",
            width=1920,
            height=1080,
            params=[1000.0, 1000.0, 960.0, 540.0]
        )
        
        dist_params = camera.get_distortion_params()
        self.assertEqual(len(dist_params), 0)
        self.assertFalse(camera.has_distortion())
        
        # OPENCV model (with distortion)
        camera = Camera(
            id=2,
            model="OPENCV",
            width=1920,
            height=1080,
            params=[1000.0, 1000.0, 960.0, 540.0, 0.1, 0.01, 0.001, 0.0001]
        )
        
        dist_params = camera.get_distortion_params()
        self.assertEqual(len(dist_params), 4)
        self.assertEqual(dist_params[0], 0.1)    # k1
        self.assertEqual(dist_params[1], 0.01)   # k2
        self.assertEqual(dist_params[2], 0.001)  # p1
        self.assertEqual(dist_params[3], 0.0001) # p2
        self.assertTrue(camera.has_distortion())


if __name__ == "__main__":
    unittest.main()
