import numpy as np
from typing import Tuple, Any, Iterator, KeysView, Tuple, TYPE_CHECKING

from .image import Image
from .camera import Camera
from .point3d import Point3D
from .types import CAMERA_MODEL_IDS

if TYPE_CHECKING:
    from .reconstruction import ColmapReconstruction

class BaseView:
    """Base class for dictionary views."""
    def __init__(self, reconstruction: 'ColmapReconstruction'):
        self._recon = reconstruction # Reference to the main reconstruction object

    def __len__(self) -> int:
        raise NotImplementedError

    def __contains__(self, key: int) -> bool:
        raise NotImplementedError

    def __getitem__(self, key: int) -> Any:
        raise NotImplementedError

    def get(self, key: int, default: Any = None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def keys(self) -> KeysView[int]:
        raise NotImplementedError

    def values(self) -> Iterator[Any]:
        raise NotImplementedError

    def items(self) -> Iterator[Tuple[int, Any]]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[int]:
        return iter(self.keys())


class CameraView(BaseView):
    """Provides dict-like access to Camera objects, created on-the-fly."""
    def __len__(self) -> int:
        return self._recon._num_cameras

    def __contains__(self, camera_id: int) -> bool:
        return camera_id in self._recon._camera_id_to_row

    def __getitem__(self, camera_id: int) -> Camera:
        if camera_id not in self._recon._camera_id_to_row:
            raise KeyError(f"Camera with ID {camera_id} not found.")
        row = self._recon._camera_id_to_row[camera_id]

        model_id = self._recon._camera_model_ids[row]
        model_name = CAMERA_MODEL_IDS[model_id].model_name
        width = self._recon._camera_widths[row]
        height = self._recon._camera_heights[row]
        params = self._recon._camera_params[row] # Already padded

        # Instantiate Camera object on-the-fly
        return Camera(id=camera_id, model=model_name, width=int(width), height=int(height), params=params)

    def keys(self) -> KeysView[int]:
        return self._recon._camera_id_to_row.keys()

    def values(self) -> Iterator[Camera]:
        # Generator yielding on-the-fly Camera objects
        return (self[key] for key in self.keys())

    def items(self) -> Iterator[Tuple[int, Camera]]:
        # Generator yielding on-the-fly (id, Camera) pairs
        return ((key, self[key]) for key in self.keys())


class ImageView(BaseView):
    """Provides dict-like access to Image objects, created on-the-fly."""
    def __len__(self) -> int:
        return self._recon._num_images

    def __contains__(self, image_id: int) -> bool:
        return image_id in self._recon._image_id_to_row

    def __getitem__(self, image_id: int) -> Image:
        if image_id not in self._recon._image_id_to_row:
            raise KeyError(f"Image with ID {image_id} not found.")
        row = self._recon._image_id_to_row[image_id]

        name = self._recon._image_names[row]
        camera_id = self._recon._image_camera_ids[row]
        qvec = self._recon._image_qvecs[row]
        tvec = self._recon._image_tvecs[row]

        # Extract features for this image
        start, end = self._recon._image_feature_indices[row]
        if start < end:
            xys = self._recon._all_xys[start:end]
            point3D_ids = self._recon._all_point3D_ids[start:end]
        else:
            xys = np.empty((0, 2), dtype=np.float64)
            point3D_ids = np.empty((0,), dtype=np.int64)

        # Instantiate Image object on-the-fly
        return Image(id=image_id, name=name, camera_id=int(camera_id),
                     qvec=qvec, tvec=tvec, xys=xys, point3D_ids=point3D_ids)

    def keys(self) -> KeysView[int]:
        return self._recon._image_id_to_row.keys()

    def values(self) -> Iterator[Image]:
        return (self[key] for key in self.keys())

    def items(self) -> Iterator[Tuple[int, Image]]:
        return ((key, self[key]) for key in self.keys())


class Point3DView(BaseView):
    """Provides dict-like access to Point3D objects, created on-the-fly."""
    def __len__(self) -> int:
        return self._recon._num_points3D

    def __contains__(self, point3D_id: int) -> bool:
        return point3D_id in self._recon._point3D_id_to_row

    def __getitem__(self, point3D_id: int) -> Point3D:
        if point3D_id not in self._recon._point3D_id_to_row:
             raise KeyError(f"Point3D with ID {point3D_id} not found.")
        row = self._recon._point3D_id_to_row[point3D_id]

        xyz = self._recon._point_xyzs[row]
        rgb = self._recon._point_rgbs[row]
        error = self._recon._point_errors[row]

        # Extract track for this point
        start, end = self._recon._point_track_indices[row]
        if start < end:
             image_ids = self._recon._all_track_image_ids[start:end]
             point2D_idxs = self._recon._all_track_point2D_idxs[start:end]
        else:
            image_ids = np.empty((0,), dtype=np.int64)
            point2D_idxs = np.empty((0,), dtype=np.int64)

        # Instantiate Point3D object on-the-fly
        return Point3D(id=point3D_id, xyz=xyz, rgb=rgb, error=float(error),
                       image_ids=image_ids, point2D_idxs=point2D_idxs)

    def keys(self) -> KeysView[int]:
        return self._recon._point3D_id_to_row.keys()

    def values(self) -> Iterator[Point3D]:
        return (self[key] for key in self.keys())

    def items(self) -> Iterator[Tuple[int, Point3D]]:
        return ((key, self[key]) for key in self.keys())


