"""This module provides utility methods for extracting PDOK BGT data."""

import json
from datetime import datetime
import time
import requests
import os
import pandas as pd
from requests.exceptions import HTTPError
import xml.etree.ElementTree as ET

from ..utils import csv_utils
from ..utils import math_utils


CITYGML_NS = {'imgeo': 'http://www.geostandaarden.nl/imgeo/2.1',
              'gml': 'http://www.opengis.net/gml'}
TIME_FMT = '%Y-%m-%dT%H:%M:%S'


def scrape_pdok_bgt(poly_list=None, bbox=None,
                    bgt_layers=["paal", "pand", "vegetatieobject"]):
    if (poly_list is None) and (bbox is None):
        print('You must provide either a polygon list or a bounding box.')
        return None
    if (poly_list is not None) and (bbox is not None):
        print('Both polygon list and bounding box provided; '
              + 'using polygon list.')
    if poly_list is None:
        poly_list = math_utils.poly_list_from_bbox(bbox)

    # Convert poly_list of points to string for PDOK request.
    poly_str = '('
    for point in poly_list:
        poly_str += f'{str(point[0])} {str(point[1])},'
    poly_str = poly_str[:-1] + ')'

    # PDOK API setup.
    base_url = 'https://api.pdok.nl'
    post_url = base_url + '/lv/bgt/download/v1_0/full/custom'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    data = {"featuretypes": bgt_layers, "format": "citygml",
            "geofilter": "POLYGON(" + poly_str + ")"}

    try:
        # Send download request.
        response = requests.post(post_url, headers=headers,
                                 data=json.dumps(data))
        response.raise_for_status()

        # Check status.
        status_url = base_url + response.json()['_links']['status']['href']
        complete = False
        while not complete:
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            complete = status_response.json()['status'] == 'COMPLETED'
            time.sleep(0.5)

        # Get download link.
        download_url = (base_url
                        + status_response.json()['_links']['download']['href'])
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

    print(f'File ready to download:\n{download_url}')
    return download_url


def parse_vegetation(vegetation_file, bbox=None, out_folder='',
                     out_file='bgt_trees.csv'):
    root = ET.parse(vegetation_file).getroot()
    trees = []
    hedges = []
    for item in root:
        for obj in item:
            plus_type = obj.find('imgeo:plus-type', CITYGML_NS).text
            bgt_status = obj.find('imgeo:bgt-status', CITYGML_NS).text
            if bgt_status == 'bestaand' and plus_type == 'boom':
                pos = obj.find('.//gml:pos', CITYGML_NS).text.split(' ')
                trees.append([plus_type, float(pos[0]), float(pos[1])])
            elif bgt_status == 'bestaand' and plus_type == 'haag':
                points = [float(x) for x
                          in obj.find('.//gml:posList',
                                      CITYGML_NS).text.split(' ')]
                polygon = list(zip(*(iter(points),) * 2))
                hedges.append([plus_type, polygon])

    if bbox is not None:
        trees = [tree for tree in trees
                 if math_utils.point_in_bbox(tree[1:3], bbox)]

    csv_headers = ['Type', 'X', 'Y']
    csv_utils.write_csv(os.path.join(out_folder, out_file), trees, csv_headers)

    # TODO iets doen met de "haag" objecten?
    # Wat kan er verder nog voorkomen in de BGT?
    return os.path.join(out_folder, out_file)


def parse_poles(pole_file, bbox=None, out_folder='', out_file='bgt_poles.csv'):
    root = ET.parse(pole_file).getroot()
    poles = []
    for item in root:
        for obj in item:
            plus_type = obj.find('imgeo:plus-type', CITYGML_NS).text
            bgt_status = obj.find('imgeo:bgt-status', CITYGML_NS).text
            if bgt_status == 'bestaand':
                pos = obj.find('.//gml:pos', CITYGML_NS).text.split(' ')
                poles.append([plus_type, float(pos[0]), float(pos[1])])

    if bbox is not None:
        poles = [pole for pole in poles
                 if math_utils.point_in_bbox(pole[1:3], bbox)]

    csv_headers = ['Type', 'X', 'Y']
    csv_utils.write_csv(os.path.join(out_folder, out_file), poles, csv_headers)
    return os.path.join(out_folder, out_file)


def parse_buildings(building_file, bbox=None, out_folder='',
                    out_file='bgt_buildings.csv'):
    root = ET.parse(building_file).getroot()
    buildings = dict()
    for obj in root:
        for building_part in obj:
            date = building_part.find('imgeo:LV-publicatiedatum',
                                      CITYGML_NS).text
            bgt_status = building_part.find('imgeo:bgt-status',
                                            CITYGML_NS).text
            bag_id = building_part.find('imgeo:identificatieBAGPND',
                                        CITYGML_NS).text
            if bag_id in buildings:
                if (datetime.strptime(date, TIME_FMT)
                        < datetime.strptime(buildings[bag_id]['Date'],
                                            TIME_FMT)):
                    continue
            if bgt_status == 'bestaand':
                ext_poly_list = []
                int_poly_list = []
                for exterior in building_part.findall('.//gml:exterior',
                                                      CITYGML_NS):
                    for pos_list in exterior.findall('.//gml:posList',
                                                     CITYGML_NS):
                        points = [float(x) for x in pos_list.text.split(' ')]
                        polygon = list(zip(*(iter(points),) * 2))
                        if (len(ext_poly_list) > 0
                                and (polygon[0] == ext_poly_list[-1])):
                            polygon = polygon[1:]
                        ext_poly_list.extend(polygon)
                    buildings[bag_id] = {'Type': 'exterior',
                                         'Date': date,
                                         'Polygon': ext_poly_list}
                for interior in building_part.findall('.//gml:interior',
                                                      CITYGML_NS):
                    for pos_list in interior.findall('.//gml:posList',
                                                     CITYGML_NS):
                        points = [float(x) for x in pos_list.text.split(' ')]
                        polygon = list(zip(*(iter(points),) * 2))
                        if (len(int_poly_list) > 0
                                and (polygon[0] == int_poly_list[-1])):
                            polygon = polygon[1:]
                        int_poly_list.extend(polygon)
                    buildings[bag_id] = {'Type': 'interior',
                                         'Date': date,
                                         'Polygon': ext_poly_list}

    if bbox is not None:
        buildings = [[k, v['Type'], v['Polygon']]
                     for (k, v) in buildings.items()
                     if math_utils.poly_overlaps_bbox(v['Polygon'], bbox)]

    csv_headers = ['BAG_ID', 'Type', 'Polygon']
    csv_utils.write_csv(os.path.join(out_folder, out_file),
                        buildings, csv_headers)
    return os.path.join(out_folder, out_file)


def merge_point_files(point_files, out_folder='', out_file='bgt_points.csv'):
    dfs = (pd.read_csv(f, sep=',') for f in point_files)
    df_merged = pd.concat(dfs, ignore_index=True)
    df_merged.to_csv(os.path.join(out_folder, out_file), index=False)
    return os.path.join(out_folder, out_file)
