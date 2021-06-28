import numpy as np

# Two libraries necessary for the CloudCompare Python wrapper
# Installation instructions in notebook [3. Clustering based region growing]
import pycc
import cccorelib


class LabelConnectedComp:
    """
    Clustering based region growing implementation using label connected comp.
    """
    def __init__(self, octree_level=9, min_component_size=100):
        """ Init variables. """
        self.octree_level = octree_level
        self.min_component_size = min_component_size

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
                        mask, threshold=0.1):
        """ When one initial seed point is found inside a component,
        make the whole component this label. """
        label_mask = np.zeros(len(mask), dtype=bool)

        pre_seed_count = np.count_nonzero(las_label == seed_point_label)

        mask_indices = np.where(mask)[0]

        cc_labels, counts = np.unique(self.point_components,
                                      return_counts=True)

        cc_labels_filtered = cc_labels[counts >= self.min_component_size]

        for cc in cc_labels_filtered:
            # select points that belong to the cluster
            cc_mask = (self.point_components == cc)
            # cluster size
            cc_size = np.count_nonzero(cc_mask)
            if cc_size < self.min_component_size:
                continue
            # number of point in teh cluster that are labelled as seed point
            seed_count = np.count_nonzero(
                las_label[mask_indices[cc_mask]] == seed_point_label)
            # at least X% of the cluster should be seed points
            if (float(seed_count) / cc_size) > threshold:
                label_mask[mask_indices[cc_mask]] = True

        # Add label to the regions
        labels = las_label
        labels[label_mask] = seed_point_label

        post_seed_count = np.count_nonzero(labels == seed_point_label)

        print(f'{post_seed_count - pre_seed_count} points added.')

        return labels
