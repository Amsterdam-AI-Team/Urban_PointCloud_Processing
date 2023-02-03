# Urban_PointCloud_Processing by Amsterdam Intelligence, GPL-3.0 license

class Labels:
    """
    Convenience class for label codes.
    """
    # TODO
    UNLABELLED = 0
    GROUND = 9
    BUILDING = 10
    TREE = 30
    STREET_LIGHT = 60
    TRAFFIC_SIGN = 62
    TRAFFIC_LIGHT = 61
    CAR = 40
    CITY_BENCH = 80
    RUBBISH_BIN = 81
    ROAD = 1
    CABLE = 79
    TRAM_CABLE = 70
    ARMATUUR = 88
    NOISE = 99

    STR_DICT = {
        0: 'Unknown',
        1: 'Road',
        2: 'Sidewalk',
        9: 'Other ground',
        10: 'Building',
        11: 'Wall (free standing)',
        12: 'Fence',
        13: 'Houseboat',
        14: 'Bus / tram shelter',
        29: 'Other structure',
        30: 'Tree',
        31: 'Potted plant',
        39: 'Other vegetation ',
        40: 'Car',
        41: 'Truck',
        42: 'Bus',
        43: 'Tram',
        44: 'Bicycle',
        45: 'Scooter / Motorcycle',
        49: 'Other vehicle',
        50: 'Person',
        51: 'Cyclist',
        60: 'Streetlight',
        61: 'Traffic light',
        62: 'Traffic sign',
        63: 'Signpost',
        64: 'Flagpole',
        65: 'Bollard',
        68: 'Complex pole',
        69: 'Other pole',
        70: 'Tram cable',
        79: 'Other cable',
        80: 'City bench',
        81: 'Rubbish bin',
        82: 'Rubbish container',
        83: 'Letter box',
        84: 'Parking meter',
        85: 'EV charging station',
        86: 'Fire hydrant',
        87: 'Bicycle rack',
        88: 'Terrace',
        89: 'Hanging streetlight',
        98: 'Other object',
        99: 'Noise'
    }

    OLD_TO_NEW = {
        1: 9,
        2: 10,
        3: 30,
        4: 60,
        5: 62,
        6: 61,
        7: 40,
        8: 80,
        9: 81,
        10: 1,
        11: 79,
        13: 70,
        14: 88,
        15: 50,
        16: 44,
        17: 65,
        18: 83,
        19: 84,
        20: 82,
        21: 31,
        22: 88,
        99: 99
    }

    @staticmethod
    def get_str(label):
        return Labels.STR_DICT[label]
