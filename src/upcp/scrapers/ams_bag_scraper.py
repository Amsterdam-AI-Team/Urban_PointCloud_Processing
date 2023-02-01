"""
This script scrapes building footprint data from the Amsterdam BAG API.
"""

import numpy as np
import json
from owslib.wfs import WebFeatureService

from ..utils.math_utils import compute_bounding_box

WFS_URL = "https://data.3dbag.nl/api/BAG3D_v2/wfs"

def scrape_amsterdam_bag(layer_name, bbox):
    '''
    layer_name: "BAG3D_v2:lod12"
    '''
    
    wfs = WebFeatureService(url=WFS_URL, version='1.1.0')

    response = wfs.getfeature(
        typename=layer_name,
        bbox=[bbox[0][0], bbox[1][1], bbox[1][0], bbox[0][1]],
        srsname='urn:x-ogc:def:crs:EPSG:28992',
        outputFormat='json'
    )

    try:
        j = json.loads(response.read().decode("utf-8"))
        return j['features']
    except ValueError:
        return None

def parse_polygons(json_response, include_bbox=True):
    """
    Parse the JSON content and transform it into a table structure.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    include_bbox : bool (default: True)
        Whether to include a bounding box for each poly.
    """
    parsed_content = []
    pand_idx = []
    name = 'pand'
    for item in json_response:
        # name = item['properties']['bgt_functie']
        polygon = item['geometry']['coordinates'][0]

        if include_bbox:
            (x_min, y_min, x_max, y_max) = compute_bounding_box(
                                                        np.array(polygon))
            parsed_content.append([name, polygon, x_min, y_max, x_max, y_min])
        else:
            parsed_content.append([name, polygon])
        pand_idx.append(item['id'])

    return parsed_content, pand_idx
