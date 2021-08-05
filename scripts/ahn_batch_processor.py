#!/usr/bin/python

import argparse
import os
import sys
import glob
from pathlib import Path
from tqdm.contrib.concurrent import process_map  # or thread_map

# Helper script to allow importing from parent folder.
import set_path  # noqa: F401
from src.preprocessing.ahn_preprocessing import process_ahn_las_tile


def _process_file(file):
    process_ahn_las_tile(file, out_folder=args.out_folder,
                         resolution=args.resolution)


if __name__ == '__main__':
    global args

    desc_str = '''This script provides batch processing of a folder of AHN LAS
                  point clouds to extract ground and building surfaces. The
                  results are saved to .npz.'''
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('--in_folder', metavar='path', action='store',
                        type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', action='store',
                        type=str, required=False)
    parser.add_argument('--resolution', metavar='float', action='store',
                        type=float, required=False, default=0.1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', metavar='int', action='store',
                        type=int, required=False, default=1)
    args = parser.parse_args()

    if args.out_folder is None:
        args.out_folder = args.in_folder

    if not os.path.isdir(args.in_folder):
        raise FileNotFoundError('The input path does not exist')

    if args.out_folder != args.in_folder:
        Path(args.out_folder).mkdir(parents=True, exist_ok=True)

    file_types = ('.LAS', '.las', '.LAZ', '.laz')
    files = [f for f in glob.glob(os.path.join(args.in_folder, '*'))
             if f.endswith(file_types)]

    if args.resume:
        # Find which files have already been processed.
        done = set([file.name[-13:-4] for file
                    in Path(args.out_folder).glob('*.npz')])
        files = [f for f in files if f[-13:-4] not in done]

    # Chunk size can be used to reduce overhead for a large number of files.
    chunk = 1
    if len(files) > 100:
        chunk = 5
    if len(files) > 1000:
        chunk = 10

    # Distribute the batch over _max_workers_ cores.
    r = process_map(_process_file, files, max_workers=args.workers,
                    chunksize=chunk)
