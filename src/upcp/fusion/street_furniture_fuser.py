# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""Street Furniture Fuser"""

import numpy as np
import logging

from ..abstract_processor import AbstractProcessor
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.math_utils import minimum_bounding_rectangle
from ..labels import Labels

logger = logging.getLogger(__name__)


class BGTStreetFurnitureFuser(AbstractProcessor):
    """
    Data Fuser class for automatic labelling of street furniture (point)
    objects such as rubbish bins and city benches using BGT data.

    Parameters
    ----------
    label : int
        Class label to use for this fuser.
    bgt_type : str
        Specify the 'type' of point object: 'bank' or 'afvalbak'
    bgt_reader : BGTPointReader object
        Used to load street furniture points.
    ahn_reader : AHNReader object
        Elevation data reader.
    """

    def __init__(self, label, bgt_type, bgt_reader, ahn_reader,
                 grid_size=0.05, min_component_size=1500,
                 padding=0, max_dist=1., params={}):
        super().__init__(label)
        self.bgt_type = bgt_type
        self.bgt_reader = bgt_reader
        self.ahn_reader = ahn_reader
        self.grid_size = grid_size
        self.min_component_size = min_component_size
        self.padding = padding
        self.max_dist = max_dist
        self.params = params

    def _label_street_furniture_like_components(self, points, ground_z,
                                                point_components, bgt_points,
                                                min_height, max_height,
                                                min_width, max_width,
                                                min_length, max_length):
        """
        Based on certain properties of street furniture objects we label
        clusters.
        """

        street_furniture_mask = np.zeros(len(points), dtype=bool)
        object_count = 0

        cc_labels = np.unique(point_components)

        cc_labels = set(cc_labels).difference((-1,))

        for cc in cc_labels:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)

            target_z = ground_z[cc_mask]
            valid_values = target_z[np.isfinite(target_z)]

            if valid_values.size != 0:
                cc_z = np.mean(valid_values)
                min_z = cc_z + min_height
                max_z = cc_z + max_height
                cluster_height = np.amax(points[cc_mask][:, 2])
                if min_z <= cluster_height <= max_z:
                    mbrect, _, mbr_width, mbr_length, center_point =\
                        minimum_bounding_rectangle(points[cc_mask][:, :2])
                    if (min_width < mbr_width < max_width and
                            min_length < mbr_length < max_length):
                        for bgt_point in bgt_points:
                            dist = np.linalg.norm(bgt_point - center_point)
                            if dist <= self.max_dist:
                                street_furniture_mask[cc_mask] = True
                                object_count += 1
                                break

        logger.debug(f'{object_count} {self.bgt_type} objects labelled.')
        return street_furniture_mask

    def get_labels(self, points, labels, mask, tilecode):
        """
        Returns the labels for the given pointcloud.

        Parameters
        ----------
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points,)
            Ignored by this class.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.

        Returns
        -------
        An array of shape (n_points,) with the updated labels.
        """
        logger.info('Street furniture fuser ' +
                    f'(label={self.label}, {Labels.get_str(self.label)}).')

        label_mask = np.zeros((len(points),), dtype=bool)

        bgt_points = self.bgt_reader.filter_tile(
                                    tilecode, bgt_types=[self.bgt_type],
                                    padding=self.padding, return_types=False)
        if len(bgt_points) == 0:
            logger.debug(f'No {self.bgt_type} objects found in tile, ' +
                         ' skipping.')
            return label_mask

        # Get the interpolated ground points of the tile
        ground_z = self.ahn_reader.interpolate(
                            tilecode, points[mask], mask, 'ground_surface')

        lcc = LabelConnectedComp(self.label, grid_size=self.grid_size,
                                 min_component_size=self.min_component_size)
        point_components = lcc.get_components(points[mask])

        # Label street_furniture like clusters
        street_furniture_mask = (self._label_street_furniture_like_components(
                                 points[mask], ground_z, point_components,
                                 bgt_points, **self.params))
        label_mask[mask] = street_furniture_mask
        labels[label_mask] = self.label

        return labels
