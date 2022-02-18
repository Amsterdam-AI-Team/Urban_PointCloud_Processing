# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""
This script scrapes building footprint data from the Amsterdam BGT API.
The documentation can be found at:
https://www.amsterdam.nl/stelselpedia/bgt-index/producten-bgt/prodspec-bgt-dgn-imgeo/
"""

import numpy as np
import requests

from ..utils.math_utils import compute_bounding_box

WFS_URL = 'https://map.data.amsterdam.nl/maps/bgtobjecten?'


def scrape_amsterdam_bgt(layer_name, bbox=None):
    """
    Scrape BGT layer information from the WFS.

    Parameters
    ----------
    layer_name : str
        Information about the different layers can be found at:
        https://www.amsterdam.nl/stelselpedia/bgt-index/producten-bgt/prodspec-bgt-dgn-imgeo/

    Returns
    -------
    The WFS response in JSON format or a dict.
    """
    params = 'REQUEST=GetFeature&' \
             'SERVICE=wfs&' \
             'VERSION=2.0.0&' \
             'TYPENAME=' \
             + layer_name + '&'

    if bbox is not None:
        bbox_string = str(bbox[0][0]) + ',' + str(bbox[0][1]) + ',' \
                      + str(bbox[1][0]) + ',' + str(bbox[1][1])
        params = params + 'BBOX=' + bbox_string + '&'

    params = params + 'OUTPUTFORMAT=geojson'

    response = requests.get(WFS_URL + params)
    try:
        return response.json()
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
    name = '_'.join(json_response['name'].split('_')[2:])
    for item in json_response['features']:
        # name = item['properties']['bgt_functie']
        polygon = item['geometry']['coordinates'][0]

        if include_bbox:
            (x_min, y_min, x_max, y_max) = compute_bounding_box(
                                                        np.array(polygon))
            parsed_content.append([name, polygon, x_min, y_max, x_max, y_min])
        else:
            parsed_content.append([name, polygon])

    return parsed_content


def parse_points_bgtplus(json_response):
    """
    Parse the JSON content and transform it into a table structure.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    """
    parsed_content = []

    for item in json_response['features']:
        name = item['properties']['plus_type']
        point = item['geometry']['coordinates']

        parsed_content.append([name, point[0], point[1]])

    return parsed_content
