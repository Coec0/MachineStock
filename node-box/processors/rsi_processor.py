from processors.node_box_processor import NodeBoxProcessor
from numpy import ndarray
from collections import deque


class RSIProcessor(NodeBoxProcessor):

    def __init__(self, window_size, use_minutes=False):
        self.last_rsi = 1
        self.window_size = window_size
        self.w = 2 / (self.window_size + 1)
        self.window = deque(maxlen=window_size)
        self.use_minutes = use_minutes
        self.time = -1
        self.open_price = 0
        self.latest_price = 0

    def process(self, timestamp, features: ndarray) -> (int, float):
        self.append_data(timestamp, features)
        sum_up = 0
        sum_dw = 0.000001
        for x in self.window:
            if x > 0:
                sum_up += x
            else:
                sum_dw += x
        rs = sum_up / abs(sum_dw)
        rsi = 100 - (100 / (1 + rs))
        return timestamp, rsi/100

    def append_data(self, timestamp, data):
        if self.time == -1:
            self.time = timestamp
        if timestamp > self.time + self.window_size*60:
            diff = self.open_price - self.latest_price
            while timestamp > self.time + 60 * self.window_size:
                self.window.append(diff)
                self.time = self.time + self.window_size * 60
            self.open_price = data[0]
        self.latest_price = data[0]
