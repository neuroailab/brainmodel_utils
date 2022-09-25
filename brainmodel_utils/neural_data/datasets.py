import numpy as np


class DatasetBase(object):
    def __init__(self, data_path, protocol=2):
        self.data_path = data_path
        self.protocol = protocol
        self.packaged_data = None

    def fetch(self):
        pass

    def save_data(self, save_path):
        assert self.packaged_data is not None
        np.savez(save_path, self.packaged_data)
