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
    debug : bool (default: False)
        Log extra debug info (optional).
    """

    def __init__(self, label, debug=False):
        self.label = label
        self.debug = debug
        self.log = ''
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

    def _log(self, message, is_debug=False, newline=True):
        """Append a message to the log."""
        if is_debug and not self.debug:
            return
        else:
            message = '[DEBUG] ' + message
        if newline:
            self.log = self.log + message + '\n'
        else:
            self.log = self.log + message

    def _debug(self, message):
        self._log(message, is_debug=True)

    def set_debug(self, debug):
        """Set the debug flag."""
        self.debug = debug

    def get_label(self):
        """Returns the label of this AbstractProcessor."""
        return self.label

    def get_log(self):
        """Returns the log of this AbstractProcessor."""
        return self.log
