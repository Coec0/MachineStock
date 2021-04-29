from processors.node_box_processor import NodeBoxProcessor
from numpy import ndarray
from collections import deque


class RSIProcessor(NodeBoxProcessor):

    def __init__(self, window_size, use_minutes=False):
        self.time_segments = window_size*60
        # 14 window_size min segments
        self.window = deque(maxlen=14)
        self.time = -1
        self.open_price = 0

    def process(self, timestamp, features: ndarray) -> (int, list):
        self.append_data(timestamp, features)
        up_sum = 0
        dw_sum = 0.000001
        for delta in self.window:
            (up_val, dw_val) = (delta, 0) if delta > 0 else (0, -delta)
            up_sum += up_val
            dw_sum += dw_val
        rs = up_sum/dw_sum
        rsi = 100 - (100 / (1 + rs))
        return timestamp, [rsi/100]

    def append_data(self, timestamp, data):
        if self.time == -1 or timestamp >= self.time + self.time_segments:
            self.time = timestamp
            diff = data[0] - self.open_price
            self.window.append(diff)
            self.open_price = data[0]
            self.time = timestamp
