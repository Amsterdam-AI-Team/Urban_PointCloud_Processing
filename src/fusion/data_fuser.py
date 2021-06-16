"""Data Reader Class"""

from abc import ABC, abstractmethod


class DataFuser(ABC):

    def __init__(self, label):
        self.label = label
        super().__init__()

    @abstractmethod
    def filter_tile(self, tilecode):
        pass

    @abstractmethod
    def get_label_mask(self, tilecode, points, mask):
        pass

    def get_label(self):
        return self.label
