"""Visualisation utilities."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon

from ..utils import las_utils
from ..utils import bgt_utils
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


def plot_bgt(tilecode, building_file=None, road_file=None, point_file=None):
    pt_labels = {'boom': 'Tree',
                 'lichtmast': 'Lamp post',
                 'verkeersbord': 'Traffic sign'}
    pt_colors = {'boom': 'green',
                 'lichtmast': 'orange',
                 'verkeersbord': 'red'}

    padding = 2.5

    buildings = []
    roads = []
    points = []

    if building_file:
        buildings = bgt_utils.get_polygons(building_file, tilecode)
    if road_file:
        roads = bgt_utils.get_polygons(road_file, tilecode)
    if point_file:
        points = bgt_utils.get_points(point_file, tilecode, padding=padding)

    ((x_min, y_max), (x_max, y_min)) =\
        las_utils.get_bbox_from_tile_code(tilecode)
    fig, ax = plt.subplots(1, constrained_layout=True)

    for poly in [Polygon(bld) for bld in buildings]:
        x, y = poly.exterior.xy
        ax.fill(x, y, c='lightblue', label='Building')
        ax.plot(x, y, c='blue')

    for poly in [Polygon(rd) for rd in roads]:
        x, y = poly.exterior.xy
        ax.fill(x, y, c='lightgrey', label='Road')

    for pt in points:
        ax.scatter(pt[1], pt[2],
                   c=pt_colors[pt[0]], marker='x', label=pt_labels[pt[0]])

    box = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=1, linestyle='--', edgecolor='black',
                            fill=False)
    ax.add_patch(box)

    ax.set_title(tilecode)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_xticks(range(x_min, x_max+1, 10))
    ax.set_xticklabels(range(x_min, x_max+1, 10))
    ax.set_yticks(range(y_min, y_max+1, 10))
    ax.set_yticklabels(range(y_min, y_max+1, 10))

    plt.xlim((x_min - padding, x_max + padding))
    plt.ylim((y_min - padding, y_max + padding))
    plt.show()


def plot_ahn_surface(ahn_tile, surf='ground_surface', ax=None):
    full_plot = False
    if ax is None:
        full_plot = True
        fig, ax = plt.subplots(1)

    x_min = int(ahn_tile['x'][0])
    x_max = int(np.ceil(ahn_tile['x'][-1]))
    y_min = int(ahn_tile['y'][-1])
    y_max = int(np.ceil(ahn_tile['y'][0]))

    im = ax.imshow(ahn_tile[surf], extent=[x_min, x_max, y_min, y_max],
                   interpolation='none')

    ax.set_title(surf)
    ax.set_xticks(range(x_min, x_max+1, 10))
    ax.set_xticklabels(range(x_min, x_max+1, 10))
    ax.set_yticks(range(y_min, y_max+1, 10))
    ax.set_yticklabels(range(y_min, y_max+1, 10))
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Elevation (m)', rotation=270, labelpad=10)
    if full_plot:
        plt.show()


def plot_ahn_sidebyside(ahn_tile):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_ahn_surface(ahn_tile, surf='ground_surface', ax=ax1)
    plot_ahn_surface(ahn_tile, surf='building_surface', ax=ax2)
    plt.tight_layout()
    plt.show()
