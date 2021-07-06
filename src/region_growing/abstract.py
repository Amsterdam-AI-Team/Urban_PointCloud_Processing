"""Data Fuser Abstract Base Class"""

from abc import ABC, abstractmethod


class AbstractRegionGrowing(ABC):
    """
    Data Fuser abstract base class for automatic labelling point clouds.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    """

    def __init__(self, label):
        self.label = label
        super().__init__()

    @abstractmethod
    def get_label_mask(self, points):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        pass

    def get_label(self):
        """Returns the label of this DataFuser object."""
        return self.label
