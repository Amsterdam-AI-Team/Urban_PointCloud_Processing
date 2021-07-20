"""
This script scrapes building footprint data from the Amsterdam BGT API.
The documentation can be found at:
https://www.amsterdam.nl/stelselpedia/bgt-index/producten-bgt/prodspec-bgt-dgn-imgeo/
"""

import requests

from ..utils.clip_utils import poly_offset
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
    return response.json()


def parse_buildings(json_response, prepare_csv=False):
    """
    Parse the JSON content and transform it into a table structure.
    Dutch-English translation of pand is building.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    """
    parsed_content = []
    for item in json_response['features']:
        pand_id = item['properties']['identificatieBAGPND']
        pand_polygon = item['geometry']['coordinates'][0]

        if prepare_csv:
            x_min, y_max, x_max, y_min = compute_bounding_box(pand_polygon)
            parsed_content.append([str(pand_id), pand_polygon, x_min, y_max,
                                  x_max, y_min])
        else:
            parsed_content.append(pand_polygon)

    csv_headers = ['building_id', 'polygon', 'x_min', 'y_max', 'x_max',
                   'y_min']

    return parsed_content, csv_headers


def parse_polygons(json_response, offset_meter=0.0, prepare_csv=False):
    """
    Parse the JSON content and transform it into a table structure.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    offset_meter : int
        Offset to inflate/deflate the polygon.
    """
    parsed_content = []
    for item in json_response['features']:
        name = item['properties']['bgt_functie']
        polygon = item['geometry']['coordinates'][0]
        polygon_w_offset = poly_offset(polygon, offset_meter)

        if prepare_csv:
            x_min, y_max, x_max, y_min = compute_bounding_box(polygon_w_offset)
            parsed_content.append([name, polygon_w_offset, x_min, y_max,
                                  x_max, y_min])
        else:
            parsed_content.append(polygon_w_offset)

    csv_headers = ['bgt_name', 'polygon', 'x_min', 'y_max', 'x_max',
                   'y_min']

    return parsed_content, csv_headers


def parse_points_bgtplus(json_response, prepare_csv=False):
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

    csv_headers = ['Type', 'X', 'Y']

    return parsed_content, csv_headers
