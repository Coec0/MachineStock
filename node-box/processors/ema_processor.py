from processors.node_box_processor import NodeBoxProcessor
from numpy import ndarray
from collections import deque


class EMAProcessor(NodeBoxProcessor):

    def __init__(self, window_size, use_minutes=False):
        self.last_ema = 1
        self.window_size = window_size
        self.w = 2 / (self.window_size + 1)
        self.window = deque(maxlen=window_size)
        self.use_minutes = use_minutes
        self.time = -1

    def process(self, timestamp, features: ndarray) -> (int, float):
        self.append_data(timestamp, features)
        ema = self.last_ema
        for p in self.window:
            ema = self.w * p + (1-self.w) * ema
        self.last_ema = ema
        return timestamp, ema

    def append_data(self, timestamp, data):
        if not self.use_minutes:
            self.window.append(data[0])
        else:
            if self.time == -1:
                self.time = timestamp
            if timestamp > self.time + 60*self.window_size:
                self.time = timestamp
                self.window.append(data[0])
