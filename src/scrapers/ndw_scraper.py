import pandas as pd
import requests
import os

URL = 'https://data.ndw.nu/api/rest/static-road-data/traffic-signs/v1/current-state?'


def get_data(town_code, out_folder='', out_file='traffic_signs.csv'):
    """
    Scrape traffic sign locations from the NDW API. The town code for
    Amsterdam is 'GM0363'.
    """
    base_url = URL + 'town-code={}'.format(town_code)

    response = requests.get(base_url)
    data = response.json()
    df = pd.json_normalize(data)

    df = df.drop_duplicates(['location.rd.x', 'location.rd.y'])
    df = df.rename(columns={'location.rd.x': 'X', 'location.rd.y': 'Y'})
    df['Type'] = 'Traffic sign'

    df.to_csv(os.path.join(out_folder, out_file), index=False)
