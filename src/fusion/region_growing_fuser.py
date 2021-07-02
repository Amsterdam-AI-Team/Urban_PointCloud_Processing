from .data_fuser import DataFuser
from ..region_growing.region_growing import RegionGrowing


class RegionGrowingFuser(DataFuser):
    """
    Region growing implementation based on:
    https://pcl.readthedocs.io/projects/tutorials/en/latest/region_growing_segmentation.html
    """
    def __init__(self, label):
        super().__init__(label)

    def get_label_mask(self, tilecode, points, mask, labels):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        seed_point_label = 2
        reg = RegionGrowing()
        reg.set_input_cloud(points, labels, seed_point_label, mask)
        labels = reg.region_growing()

        return labels
