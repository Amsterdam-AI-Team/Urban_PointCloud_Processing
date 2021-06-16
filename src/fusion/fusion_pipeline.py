"""Data Fusion Pipeline"""

import numpy as np
import os
import pathlib
from tqdm import tqdm

from ..utils.las_utils import (get_tilecode_from_filename,
                               read_las,
                               label_and_save_las)


class FusionPipeline:

    def __init__(self, fusers):
        self.fusers = fusers

    def process_cloud(self, tilecode, points, mask=None):
        if mask is None:
            mask = np.full((len(points['x']),), True)

        labels = np.zeros((len(points['x']),), dtype='uint16')
        for fuser in self.fusers:
            label_mask = fuser.get_label_mask(tilecode, points, mask)
            labels[label_mask] = fuser.get_label()
            mask[label_mask] = False

    def process_file(self, filename, outfile=None, mask=None):
        if not os.path.isfile(filename):
            print('The input file specified does not exist')
            return None

        if outfile is None:
            outfile = filename

        tilecode = get_tilecode_from_filename(filename)
        pointcloud = read_las(filename)
        points = {'x': pointcloud.x, 'y': pointcloud.y, 'z': pointcloud.z}

        if mask is None:
            mask = np.full((len(points['x']),), True)

        labels = self.process_cloud(tilecode, points, mask)
        label_and_save_las(pointcloud, labels, outfile)

    def process_folder(self, in_folder, out_folder=None, suffix='',
                       hide_progress=False):
        if not os.path.isfolder(in_folder):
            print('The input path specified does not exist')
            return None
        if type(in_folder) == str:
            in_folder = pathlib.Path(in_folder)
        if out_folder is None:
            out_folder = in_folder
        else:
            pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

        file_types = ('.LAS', '.las', '.LAZ', '.laz')

        files = [f for f in in_folder.glob('*') if f.endswith(file_types)]

        for file in tqdm(files, unit="file", disable=hide_progress):
            # Load LAS file.
            filename, extension = os.path.splitext(file)
            outfile = filename + suffix + extension
            self.process_file(file, outfile)
