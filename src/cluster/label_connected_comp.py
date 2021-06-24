import numpy as np
import pandas as pd

# CloudCompare Python wrapper libraries
# Installation inctrsuctions in notebook [3. Label connected comp]
import pycc
import cccorelib


class LabelConnectedComp:
    def __init__(self, octree_level=9):
        """ Init variables. """
        self.octree_level = octree_level

    def set_input_cloud(self, las, mask):
        """ Function to convert to CloudCompare point cloud. """
        # Be aware that CloudCompare stores coordinates on 32 bit floats.
        # To avoid losing too much precision you should 'shift' your
        # coordinates if they are 64 bit floats (which is the default in
        # python land)
        xs = (las.x[mask] - las.header.x_min).astype(pycc.PointCoordinateType)
        ys = (las.y[mask] - las.header.y_min).astype(pycc.PointCoordinateType)
        zs = (las.z[mask] - las.header.z_min).astype(pycc.PointCoordinateType)
        point_cloud = pycc.ccPointCloud(xs, ys, zs)
        # Add the global shift to CloudCompare so that it can use it,
        # for example to display the real coordinates in point picking tool
        point_cloud.setGlobalShift(-las.header.x_min, -las.header.y_min,
                                   -las.header.z_min)

        # (Optional) Create (if it does not exists already)
        # a scalar field where we store the Labels
        labels_sf_idx = point_cloud.getScalarFieldIndexByName('Labels')
        if labels_sf_idx == -1:
            labels_sf_idx = point_cloud.addScalarField('Labels')
        point_cloud.setCurrentScalarField(labels_sf_idx)

        # You can access the x,y,z fields using self.point_cloud.points()

        self.labels_sf_idx = labels_sf_idx
        self.point_cloud = point_cloud

    def label_connected_comp(self):
        """ Perform Label Connected Components. """
        component_count = (cccorelib.AutoSegmentationTools
                           .labelConnectedComponents(self.point_cloud,
                                                     level=self.octree_level))
        print('There are {} components found'.format(component_count))

        # Get the scalar field with labels and points coords as numpy array
        labels_sf = self.point_cloud.getScalarField(self.labels_sf_idx)
        self.point_components = labels_sf.asArray()

    def fill_components(self, las_label, unlabelled_label, seed_point_label,
                        mask):
        """ When one initial seed point is found inside a component,
        make the whole component this label. """
        seed_list = las_label.astype("float")[mask]
        seed_list[seed_list == unlabelled_label] = np.nan
        dataset = pd.DataFrame({'labels': self.point_components,
                               'seed_points': seed_list})
        dataset['seed_points'] = (dataset.groupby(['labels'],
                                  sort=False)['seed_points']
                                  .apply(lambda x: x.ffill().bfill()))

        seed_list_grown = (dataset['seed_points'].fillna(unlabelled_label)
                           .to_numpy())

        label_mask = np.zeros(len(mask), dtype=bool)
        mask_indices = np.where(mask)[0]

        label_mask[mask_indices[seed_list_grown == seed_point_label]] = True

        # Add label to the regions
        labels = las_label
        labels[label_mask] = seed_point_label

        print('Filled components with labels based on initial seed points')

        return labels

    def min_points_per_component(self, noise_label, min_component_size=1000):
        """ Ignore the components with less than a specified number of points.
        Similar to the option "Min. points per component" in CloudCompare. """
        all_labels, counts = np.unique(self.point_components,
                                       return_counts=True)
        small_labels = all_labels[counts < min_component_size]
        self.point_components[np.in1d(self.point_components,
                                      small_labels)] = noise_label
