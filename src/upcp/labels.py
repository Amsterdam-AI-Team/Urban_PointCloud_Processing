# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

class Labels:
    """
    Convenience class for label codes.
    """
    # Classes available for fusers
    UNKNOWN = 0
    ROAD = 1
    GROUND = 9
    BUILDING = 10
    TREE = 30
    CAR = 40
    STREET_LIGHT = 60
    TRAFFIC_LIGHT = 61
    TRAFFIC_SIGN = 62
    TRAM_CABLE = 70
    CABLE = 79
    CITY_BENCH = 80
    RUBBISH_BIN = 81
    ARMATUUR = 90
    NOISE = 99

    STR_DICT = {
        0: 'Unknown',
        1: 'Road',
        2: 'Sidewalk',
        9: 'Other ground',
        10: 'Building',
        11: 'Wall',
        12: 'Fence',
        13: 'Houseboat',
        14: 'Bridge',
        15: 'Bus/tram shelter',
        16: 'Advertising column',
        17: 'Kiosk',
        29: 'Other structure',
        30: 'Tree',
        31: 'Potted plant',
        39: 'Other vegetation ',
        40: 'Car',
        41: 'Truck',
        42: 'Bus',
        43: 'Tram',
        44: 'Bicycle',
        45: 'Scooter/Motorcycle',
        49: 'Other vehicle',
        50: 'Person',
        51: 'Person sitting',
        52: 'Cyclist',
        59: 'Other Person',
        60: 'Streetlight',
        61: 'Traffic light',
        62: 'Traffic sign',
        63: 'Signpost',
        64: 'Flagpole',
        65: 'Bollard',
        66: 'Parasol',
        68: 'Complex pole',
        69: 'Other pole',
        70: 'Tram cable',
        79: 'Other cable',
        80: 'City bench',
        81: 'Rubbish bin',
        82: 'Small container',
        83: 'Large container',
        84: 'Letter box',
        85: 'Parking meter',
        86: 'EV charging station',
        87: 'Fire hydrant',
        88: 'Bicycle rack',
        89: 'Advertising sign',
        90: 'Hanging streetlight',
        91: 'Terrace',
        92: 'Playgorund',
        93: 'Electrical box',
        94: 'Concrete block',
        95: 'Construction sign',
        98: 'Other object',
        99: 'Noise'
    }

    OLD_TO_NEW = {
        0: UNKNOWN,
        1: GROUND,
        2: BUILDING,
        3: TREE,
        4: STREET_LIGHT,
        5: TRAFFIC_SIGN,
        6: TRAFFIC_LIGHT,
        7: CAR,
        8: CITY_BENCH,
        9: RUBBISH_BIN,
        10: ROAD,
        11: CABLE,
        13: TRAM_CABLE,
        14: ARMATUUR,
        99: NOISE
    }

    @staticmethod
    def get_str(label):
        return Labels.STR_DICT[label]
