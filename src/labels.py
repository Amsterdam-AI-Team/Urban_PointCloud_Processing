# TODO rethink labelling, e.g. hierarchical?
class Labels:
    """
    Convenience class for label codes.
    """
    UNLABELLED = 0
    GROUND = 1
    BUILDING = 2
    TREE = 3
    STREET_LIGHT = 4
    TRAFFIC_SIGN = 5
    TRAFFIC_LIGHT = 6
    CAR = 7
    CITY_BENCH = 8
    RUBBISH_BIN = 9
    NOISE = 99

    STR_DICT = {0: 'Unlabelled',
                1: 'Ground',
                2: 'Building',
                3: 'Tree',
                4: 'Street light',
                5: 'Traffic sign',
                6: 'Traffic light',
                7: 'Car',
                8: 'City bench',
                9: 'Rubbish bin',
                99: 'Noise'}

    @staticmethod
    def get_str(label):
        return Labels.STR_DICT[label]
