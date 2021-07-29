"""Visualisation utilities."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..utils import las_utils
from ..utils.labels import Labels
from ..utils.interpolation import FastGridInterpolator


def plot_cloud_slice(las_file, ahn_reader, plane_height=1.5):
    las = las_utils.read_las(las_file)
    labels = las.label
    points = np.vstack((las.x, las.y, las.z)).T

    tilecode = las_utils.get_tilecode_from_filename(las_file)
    ((x_min, y_max), (x_max, y_min)) =\
        las_utils.get_bbox_from_tile_code(tilecode)
    ahn_tile = ahn_reader.filter_tile(tilecode)
    fast_z = FastGridInterpolator(ahn_tile['x'], ahn_tile['y'],
                                  ahn_tile['ground_surface'])
    points_z = fast_z(points[:, 0:2])

    plane_mask = ((points[:, 2] >= points_z + plane_height - 0.05)
                  & (points[:, 2] <= points_z + plane_height + 0.05))
    label_set = np.unique(labels[plane_mask])

    fig, ax = plt.subplots(1, constrained_layout=True)

    for label in label_set:
        label_mask = plane_mask & (labels == label)
        ax.scatter(points[label_mask, 0], points[label_mask, 1],
                   marker='.', label=Labels.get_str(label))

    box = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=1, linestyle='--', edgecolor='grey',
                            fill=False)

    ax.add_patch(box)
    ax.set_title(tilecode)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.axis('equal')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
