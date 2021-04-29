from processors.node_box_processor import NodeBoxProcessor
from numpy import ndarray
from collections import deque


class OBVProcessor(NodeBoxProcessor):

    def __init__(self, segment_length):
        self.obv = 1
        self.close_prev = 0
        self.seg_len = segment_length
        self.time = -1

    def process(self, timestamp, features: ndarray) -> (int, list):
        price = features[0]
        volume = features[1]
        if self.time == -1 or timestamp >= self.time + self.seg_len*60:
            self.time = timestamp
            if price > self.close_prev:
                self.obv = self.obv + volume
            elif price < self.close_prev:
                self.obv = self.obv - volume
            self.close_prev = price
        return timestamp, [self.obv]
