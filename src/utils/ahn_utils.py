"""This module provides utility methods for AHN data."""

import numpy as np


def load_ahn_tile(ahn_file):
    """
    Load the ground and building surface grids in a given AHN .npz file and
    return the results as a dict with keys 'x', 'y', 'ground_surface' and
    'building_surface'.
    """
    ahn = np.load(ahn_file)
    ahn_tile = {'x': ahn['x'],
                'y': ahn['y'],
                'ground_surface': ahn['ground'],
                'building_surface': ahn['building']}
    return ahn_tile
