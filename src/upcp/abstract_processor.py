# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""Abstract base class for PipeLine processor objects."""

from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """
    Abstract base class for automatic labelling point clouds. Objects of this
    class can be used in the processing PipeLine.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    """

    def __init__(self, label):
        self.label = label
        super().__init__()

    @abstractmethod
    def get_label_mask(self, points, labels, mask, tilecode):
        """
        Returns the label mask for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points,)
            The labels corresponding to each point.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.

        Returns
        -------
        An array of shape (n_points,) with dtype=bool indicating which points
        should be labelled according to this fuser.
        """
        pass

    def get_label(self):
        """Returns the label of this AbstractProcessor."""
        return self.label
