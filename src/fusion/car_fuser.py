import numpy as np
import os
from shapely.geometry import Polygon

from .abstract import AbstractFuser
from ..region_growing.label_connected_comp import LabelConnectedComp
from ..utils.interpolation import FastGridInterpolator
from ..utils.math_utils import minimum_bounding_rectangle
from ..utils.ahn_utils import load_ahn_tile


class CarFuser(AbstractFuser):  # TODO of wel gewoon abstractfuser
    def __init__(self, label, exclude_labels, data_folder, octree_level,
                 min_component_size, max_above_ground=3,
                 bgt_file=None, bgt_road_polygons=None,
                 min_width_thresh=1.5, max_width_thresh=2.55,
                 min_length_thresh=2.0, max_length_thresh=7.0):
        super().__init__(label)
        self.octree_level = octree_level
        self.min_component_size = min_component_size

        self.exclude_labels = exclude_labels
        self.data_folder = data_folder

        self.max_above_ground = max_above_ground
        self.min_width_thresh = min_width_thresh
        self.max_width_thresh = max_width_thresh
        self.min_length_thresh = min_length_thresh
        self.max_length_thresh = max_length_thresh

        if bgt_file is not None:
            print('TODO this part, get all polygons for a tile')
        elif bgt_road_polygons is not None:
            self.bgt_road_polygons = bgt_road_polygons
        else:
            print('No data file or list content specified. Aborting...')
            return None

    def _filter_tile(self, tilecode):
        """
        Returns an AHN tile dict for the area represented by the given
        CycloMedia tile-code. TODO also implement geotiff?
        """
        return load_ahn_tile(os.path.join(self.data_folder, 'ahn_' + tilecode
                                          + '.npz'))

    def _fill_car_like_components(self, fast_z, maskje, point_components,
                                  min_component_size, points):
        """ Label car like clusters.  """  # TODO text

        mask_indices = np.where(maskje)[0]
        label_mask = np.zeros(len(maskje), dtype=bool)

        cc_labels, counts = np.unique(point_components,
                                      return_counts=True)

        cc_labels_filtered = cc_labels[counts >= min_component_size]

        for cc in cc_labels_filtered:
            # select points that belong to the cluster
            cc_mask = (point_components == cc)

            target_z = fast_z(points[mask_indices[cc_mask]])
            valid_values = target_z[np.isfinite(target_z)]

            if valid_values.size != 0:
                max_z_thresh = np.mean(valid_values) + self.max_above_ground

                max_z = np.amax(points[mask_indices[cc_mask]][:, 2])  # TODO miss de cc cloud gebruiken?
                if max_z < max_z_thresh:
                    hull_points, mbr_width, mbr_length =\
                        minimum_bounding_rectangle(
                            points[mask_indices[cc_mask]][:, :2])

                    if (self.min_width_thresh < mbr_width <
                            self.max_width_thresh and self.min_length_thresh <
                            mbr_length < self.max_length_thresh):
                        p1 = Polygon(hull_points)
                        for road_polygon in self.bgt_road_polygons:
                            p2 = Polygon(road_polygon)

                            do_overlap = p1.intersects(p2)
                            if do_overlap:
                                label_mask[mask_indices[cc_mask]] = True
                                break

        return label_mask

    def get_label_mask(self, tilecode, points, mask, las_labels):
        """TODO"""
        # Get the interpolated ground points of the tile
        ahn_tile = self._filter_tile(tilecode)
        surface = ahn_tile['ground_surface']
        fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'], surface)
        
        # Create lcc object and perform lcc
        lcc = LabelConnectedComp(self.label, self.exclude_labels,
                                 octree_level=self.octree_level,
                                 min_component_size=self.min_component_size)
        lcc.perform_lcc_tasks(points, las_labels)
        
        # Label car like clusters 
        label_mask = self._fill_car_like_components(fast_z, lcc.mask,
                                                    lcc.point_components,
                                                    lcc.min_component_size,
                                                    points)

        return label_mask

