import math
from processors.node_box_processor import NodeBoxProcessor
from processors.ema_processor import EMAProcessor
from numpy import ndarray


class KlingerProcessor(NodeBoxProcessor):

    def __init__(self):
        self.vf_ema_34 = VolumeForceEMA(34)
        self.vf_ema_55 = VolumeForceEMA(55)

    def process(self, timestamp, features: ndarray) -> (int, list):
        _, vf_ema_34 = self.vf_ema_34.process(timestamp, features)
        _, vf_ema_55 = self.vf_ema_55.process(timestamp, features)
        return timestamp, [vf_ema_34[0] - vf_ema_55[0]]


class VolumeForceEMA:
    def __init__(self, window_size):
        self.ema = EMAProcessor(window_size, True)
        self.high = -1
        self.low = math.inf
        self.close = 0
        self.time = -1
        self.vf = 0
        self.hlc_prev = 0
        self.cm_prev = 0.1
        self.dm_prev = 0.1
        self.trend_prev = 0

    def process(self, timestamp, features):
        price = features[0]
        volume = features[1]
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        if self.time == -1 or timestamp >= self.time + 60:
            self.time = timestamp
            self.close = price
            self.vf = self.calc_volume_force(volume)
            self.high = -1
            self.low = math.inf
        return self.ema.process(timestamp, [self.vf])

    def calc_trend(self):
        high_low_close = self.high + self.low + self.close
        trend = 1 if high_low_close > self.hlc_prev else -1
        self.hlc_prev = high_low_close
        return trend

    def calc_volume_force(self, volume):
        trend = self.calc_trend()
        dm = self.high - self.low
        cm = self.cm_prev + dm if trend == self.trend_prev else self.dm_prev + dm
        cm = cm + 0.000001
        self.dm_prev = dm
        self.cm_prev = cm
        self.trend_prev = trend
        return volume * (2*(dm/cm-1)) * trend * 100
