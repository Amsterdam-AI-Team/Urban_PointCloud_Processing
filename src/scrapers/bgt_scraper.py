"""
This script scrapes building footprint data from the BGT API.
The documentation can be found at:
https://www.amsterdam.nl/stelselpedia/bgt-index/producten-bgt/prodspec-bgt-dgn-imgeo/
"""

import os
import requests

from ..utils.csv_utils import write_csv
from ..utils.math_utils import compute_bounding_box

WFS_URL = 'https://map.data.amsterdam.nl/maps/bgtobjecten?'


def scrape_amsterdam_bgt(layer_name, bbox=None):
    """
    Scrape BGT layer information from the WFS.

    Parameters
    ----------
    bbox_region : str (with a list inside)
        Minimum and maximum dimensions of the search space.

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


def parse_buildings(json_response, out_folder='',
                    out_file='bgt_buildings.csv'):
    """
    Parse the JSON content and transform it into a table structure.
    Dutch-English translation of pand is building.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    """
    output_list = []

    for item in json_response['features']:
        pand_id = item['properties']['identificatieBAGPND']
        pand_polygon = item['geometry']['coordinates'][0]
        x_min, y_max, x_max, y_min = compute_bounding_box(pand_polygon)

        output_list.append([str(pand_id), pand_polygon, x_min, y_max, x_max,
                           y_min])

    csv_headers = ['building_id', 'polygon', 'x_min', 'y_max', 'x_max',
                   'y_min']
    write_csv(os.path.join(out_folder, out_file), output_list, csv_headers)


def parse_points_bgtplus(json_response, out_folder='',
                         out_file='bgt_points.csv'):
    """
    Parse the JSON content and transform it into a table structure.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    """
    output_list = []

    for item in json_response['features']:
        name = item['properties']['plus_type'][0]
        point = item['geometry']['coordinates']

        output_list.append([name, point[0], point[1]])

    csv_headers = ['Type', 'X', 'Y']
    write_csv(os.path.join(out_folder, out_file), output_list, csv_headers)
