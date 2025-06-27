import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data: np.ndarray):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.max - self.min + 1e-8) + self.min 