import unittest
import numpy as np

from pycolmap.utils import (
    qvec2rotmat,
    rotmat2qvec,
    angle_between_rays,
    triangulate_point
)


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_qvec_rotmat_conversion(self):
        """Test conversion between quaternion and rotation matrix."""
        # Identity quaternion
        qvec = (1.0, 0.0, 0.0, 0.0)
        R = qvec2rotmat(qvec)
        
        # Check if result is identity matrix
        np.testing.assert_allclose(R, np.eye(3), rtol=1e-5)
        
        # Convert back to quaternion
        qvec_back = rotmat2qvec(R)
        
        # Check if original quaternion is recovered
        # Note: quaternions with q and -q represent the same rotation
        if qvec_back[0] < 0:
            qvec_back = tuple(-q for q in qvec_back)
        np.testing.assert_allclose(qvec, qvec_back, rtol=1e-5)
        
        # Test with non-identity rotation
        qvec = (0.7071, 0.7071, 0.0, 0.0)  # 90 degrees around X
        R = qvec2rotmat(qvec)
        
        # Expected rotation matrix for 90 degrees around X
        R_expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        np.testing.assert_allclose(R, R_expected, rtol=1e-5)
    
    def test_angle_between_rays(self):
        """Test calculation of angle between camera rays."""
        # Two cameras looking at a point from opposite sides (180 degrees)
        camera1 = (-1.0, 0.0, 0.0)
        camera2 = (1.0, 0.0, 0.0)
        point = (0.0, 0.0, 0.0)
        
        angle = angle_between_rays(camera1, camera2, point)
        self.assertAlmostEqual(angle, 180.0, places=5)
        
        # Two cameras looking at a point from 90 degrees
        camera1 = (0.0, 0.0, 0.0)
        camera2 = (1.0, 0.0, 0.0)
        point = (1.0, 0.0, 1.0)
        
        angle = angle_between_rays(camera1, camera2, point)
        self.assertAlmostEqual(angle, 45.0, places=5)
    
    def test_triangulate_point(self):
        """Test triangulation of a 3D point from two 2D observations."""
        # Create two simple projection matrices
        P1 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        P2 = np.array([
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        
        # A point at (2, 3, 4) should project to (2/4, 3/4) in first camera
        # and ((2+1)/4, 3/4) in second camera
        point_3d_gt = (2.0, 3.0, 4.0)
        point_2d_1 = (point_3d_gt[0] / point_3d_gt[2], point_3d_gt[1] / point_3d_gt[2])
        point_2d_2 = ((point_3d_gt[0] + 1.0) / point_3d_gt[2], point_3d_gt[1] / point_3d_gt[2])
        
        # Triangulate
        point_3d = triangulate_point(P1, P2, point_2d_1, point_2d_2)
        
        # Check result
        np.testing.assert_allclose(point_3d, point_3d_gt, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
