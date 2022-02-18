# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

"""
This script scrapes street sign data from the National Road Traffic Data Portal
(NDW) to enrich the data from Amsterdam BGT API.
"""

import requests
import numpy as np

URL = 'https://data.ndw.nu/api/rest/static-road-data/traffic-signs/v1/current-state?'


def scrape_ndw(town_code='GM0363'):
    """
    Scrape traffic sign locations from the NDW API.

    Parameters
    ----------
    town_code : str
        Municipality code.

    Returns
    -------
    The API response in JSON format or a dict.
    """
    base_url = URL + 'town-code={}'.format(town_code)

    response = requests.get(base_url)
    return response.json()


def parse_traffic_signs(json_response, bbox=None):
    """
    Parse the JSON content and transform it into a table structure.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.
    """
    name = 'verkeersbord'
    parsed_content = []

    for item in json_response:
        point = item['location']['rd']

        parsed_content.append([name, float(point['x']), float(point['y'])])

    if bbox is not None:
        # Filter for points inside the bbox
        ((bx_min, by_max), (bx_max, by_min)) = bbox
        parsed_content = np.array(parsed_content)
        X = parsed_content[:, 1].astype(float)
        Y = parsed_content[:, 2].astype(float)

        mask = (X < bx_max) & (X > bx_min) & (Y < by_max) & (Y > by_min)
        parsed_content = parsed_content[np.where(mask)].tolist()

    return parsed_content
