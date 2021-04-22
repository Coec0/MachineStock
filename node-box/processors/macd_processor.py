from node_box_processor import NodeBoxProcessor
from numpy import ndarray
from processors.ema_processor import EMAProcessor


class MACDProcessor(NodeBoxProcessor):

    def __init__(self):
        self.ema12 = EMAProcessor(12, True)
        self.ema26 = EMAProcessor(26, True)

    def process(self, timestamp, features: ndarray) -> (int, float):
        _, ema12 = self.ema12.process(timestamp, features)
        _, ema26 = self.ema26.process(timestamp, features)
        return timestamp, ema12-ema26
