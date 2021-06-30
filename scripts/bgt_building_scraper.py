#!/usr/bin/python

"""
This script scrapes building footprint data from the BGT API.
The documentation can be found at:
https://www.amsterdam.nl/stelselpedia/bgt-index/producten-bgt/prodspec-bgt-dgn-imgeo/
"""

import argparse
import os
import requests
from pathlib import Path

# Helper script to allow importing from parent folder.
import set_path  # noqa: F401
from src.utils.csv_utils import write_csv
from src.utils.math_utils import compute_bounding_box

WFS_URL = 'https://map.data.amsterdam.nl/maps/bgtobjecten?'
LAYER_NAME = 'BGT_PND_pand'


def get_WFS_features_in_bbox(bbox_region):
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
             'VERSION=1.0.0&' \
             'TYPENAME=' \
             + LAYER_NAME + '&' \
             'BBOX=' \
             + bbox_region + '&' \
             'OUTPUTFORMAT=geojson'

    response = requests.get(WFS_URL + params)
    return response.json()


def parse_BGT_data(json_response):
    """
    Parse the JSON content and transform it into a table structure.
    Dutch-English translation of pand is building.

    Parameters
    ----------
    json_response : dict
        JSON response from a WFS request.

    Returns
    -------
    A list with the pand id, pand polygon and pand bbox data.
    """
    output_list = []

    for item in json_response['features']:
        pand_id = item['properties']['identificatieBAGPND']
        pand_polygon = item['geometry']['coordinates'][0]
        ((x_min, y_max), (x_max, y_min)) = compute_bounding_box(pand_polygon)

        output_list.append([str(pand_id), pand_polygon, x_min, y_max, x_max,
                           y_min])

    return output_list


def main():
    desc_str = '''This script scrapes building footprint data, in the form of
                  polygons, from BGT. Input for bbox must be a string with four
                  comma-seperated coordinate values (x_min,y_max,x_max,y_min);
                  e.g. "110000.0,493750.0,115000.0,487500.0"'''
    parser = argparse.ArgumentParser(description=desc_str)
    parser.add_argument('--bbox_region', type=str, required=True)
    parser.add_argument('--out_folder', metavar='path', type=str,
                        required=True)
    args = parser.parse_args()

    Path(args.out_folder).mkdir(parents=True, exist_ok=True)

    bbox_region = args.bbox_region

    # Scrape the BGT data.
    json_response = get_WFS_features_in_bbox(bbox_region)
    bgt_content = parse_BGT_data(json_response)

    csv_headers = ['pand_id', 'pand_polygon', 'x_min', 'y_max', 'x_max',
                   'y_min']
    write_csv(os.path.join(args.out_folder, 'pand_scrape_' + bbox_region +
              '.csv'), bgt_content, csv_headers)

    print('Scraped BGT building footprint data inside bbox region {0}'
          .format(bbox_region))


if __name__ == "__main__":
    main()
