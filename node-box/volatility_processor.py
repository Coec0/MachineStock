import math

from node_box_processor import NodeBoxProcessor
from numpy import ndarray
from collections import deque


class VolatilityProcessor(NodeBoxProcessor):

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = deque(window_size)
        self.mean = 1
        self.var_sum = 1

    def process(self, timestamp, features: ndarray) -> (int, float):
        new_x = features[0]
        if len(self.window) == 0:
            old_x = new_x + 0.00001
            old_mean = old_x
        else:
            old_x = self.window[0]
            old_mean = self.mean

        self.mean = old_mean + ((new_x - old_x) / self.window_size)
        self.var_sum = self.var_sum + (new_x + old_x - old_mean - self.mean) * (new_x - old_x)
        self.var_sum = max(0, self.var_sum)

        self.window.append(new_x)
        volatility = math.sqrt(self.var_sum / (self.window_size - 1))
        return timestamp, volatility
