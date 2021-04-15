from numpy import ndarray
from torch import float32


class NodeBoxProcessor:
    def process(self, features: ndarray) -> (int, float32):
        """Process the data and return the result as a tuple of (timestamp, result).
        The timestamp is the timestamp of when the result is predicted for """
        pass
