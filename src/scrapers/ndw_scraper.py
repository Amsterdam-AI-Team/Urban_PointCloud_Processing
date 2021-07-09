import pandas as pd
import requests
import os

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


def parse_traffic_signs(json_response, out_folder='',
                        out_file='traffic_signs.csv'):
    """ Parse the JSON content and transform it into a table structure. """
    df = pd.json_normalize(json_response)

    df = df.drop_duplicates(['location.rd.x', 'location.rd.y'])
    df = df.rename(columns={'location.rd.x': 'X', 'location.rd.y': 'Y'})
    df['Type'] = 'Traffic sign'

    df.to_csv(os.path.join(out_folder, out_file), index=False)
