import numpy as np

from ..labels import Labels


def get_label_stats(labels):
    """Returns a string describing statistics based on labels."""
    N = len(labels)
    labels, counts = np.unique(labels, return_counts=True)
    stats = f'Total: {N:25} points\n'
    for label, cnt in zip(labels, counts):
        name = Labels.get_str(label)
        perc = (float(cnt) / N) * 100
        stats += f'Class {label:2}, {name:14} ' +\
                 f'{cnt:7} points ({perc:4.1f} %)\n'
    return stats
