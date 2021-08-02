"""Data Fusion Pipeline"""

import numpy as np
import os
import pathlib
from tqdm import tqdm

from .utils.las_utils import (get_tilecode_from_filename, read_las,
                              label_and_save_las)


class Pipeline:
    """
    Pipeline for data fusion. The class accepts a list of DataFuser objects and
    processes a single point cloud or a folder of pointclouds by applying the
    given DataFusers consecutively. It is assumed that the fusers are ordered
    by importance: points labelled by each fuser are excluded from further
    processing.

    Parameters
    ----------
    process_sequence : iterable of type AbstractProcessor
        The processors to apply, in order.
    exclude_labels : list
        List of labels to exclude from processing.
    """

    FILE_TYPES = ('.LAS', '.las', '.LAZ', '.laz')

    def __init__(self, process_sequence=[], exclude_labels=[]):
        self.process_sequence = process_sequence
        self.exclude_labels = exclude_labels

    def _create_mask(self, mask, labels):
        """Create mask based on `exclude_labels`."""
        if mask is None:
            mask = np.ones((len(labels),), dtype=bool)
        if len(self.exclude_labels) > 0:
            for exclude_label in self.exclude_labels:
                mask = mask & (labels != exclude_label)
        return mask

    def process_cloud(self, tilecode, points, labels, mask=None):
        """
        Process a single point cloud.

        Parameters
        ----------
        tilecode : str
            The CycloMedia tile-code for the given pointcloud.
        points : array of shape (n_points, 3)
            The point cloud <x, y, z>.
        labels : array of shape (n_points, 1)
            All labels as int values
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.

        Returns
        -------
        An array of shape (n_points,) with dtype=uint16 indicating the label
        for each point.
        """
        mask = self._create_mask(mask, labels)

        for obj in self.process_sequence:
            label_mask = obj.get_label_mask(points, labels, mask, tilecode)
            labels[label_mask] = obj.get_label()
            mask[label_mask] = False

        return labels

    def process_file(self, in_file, out_file=None, mask=None):
        """
        Process a single LAS file and save the result as .laz file.

        Parameters
        ----------
        in_file : str
            The file to process.
        out_file : str (default: None)
            The name of the output file. If None, the input will be
            overwritten.
        mask : array of shape (n_points,) with dtype=bool
            Pre-mask used to label only a subset of the points.
        """
        if not os.path.isfile(in_file):
            print('The input file specified does not exist')
            return None

        if out_file is None:
            out_file = in_file

        tilecode = get_tilecode_from_filename(in_file)
        pointcloud = read_las(in_file)
        points = np.vstack((pointcloud.x, pointcloud.y, pointcloud.z)).T

        if 'label' not in pointcloud.point_format.extra_dimension_names:
            labels = np.zeros((len(points),), dtype='uint16')
        else:
            labels = pointcloud.label

        labels = self.process_cloud(tilecode, points, labels, mask)
        label_and_save_las(pointcloud, labels, out_file)

    def process_folder(self, in_folder, out_folder=None, in_prefix='',
                       out_prefix='', suffix='', hide_progress=False):
        """
        Process a folder of LAS files and save each processed file.

        Parameters
        ----------
        in_folder : str or Path
           The input folder.
        out_folder : str or Path (default: None)
           The name of the output folder. If None, the output will be written
           to the input folder.
        in_prefix : str
            Optional prefix to filter files in the input folder. Only files
            starting with this prefix will be processed.
        out_prefix : str
            Optional prefix to prepend to output files. If an in_prefix is
            given, it will be replaced by the out_prefix.
        suffix : str or None
            Suffix to add to the filename of processed files. A value of None
            indicates that the same filename is kept; when out_folder=None this
            means each file will be overwritten.
        """
        if not os.path.isdir(in_folder):
            print('The input path specified does not exist')
            return None
        if type(in_folder) == str:
            in_folder = pathlib.Path(in_folder)
        if out_folder is None:
            out_folder = in_folder
        else:
            pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)
        if suffix is None:
            suffix = ''

        files = [f for f in in_folder.glob('*')
                 if f.name.endswith(self.FILE_TYPES)
                 and f.name.startswith(in_prefix)]
        files_tqdm = tqdm(files, unit="file", disable=hide_progress)

        for file in files_tqdm:
            files_tqdm.set_postfix_str(file.name)
            filename, extension = os.path.splitext(file.name)
            if in_prefix and out_prefix:
                filename = filename.replace(in_prefix, out_prefix)
            elif out_prefix:
                filename = out_prefix + filename
            outfile = out_folder + '/' + filename + suffix + extension
            self.process_file(file.as_posix(), outfile)
