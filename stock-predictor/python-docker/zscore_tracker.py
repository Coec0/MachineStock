from collections import deque
from datetime import datetime
from datetime import timedelta
from statistics import stdev
import numpy as np


class ZScoreTracker:
    def __init__(self, norm_window_size, params):
        self.window_size = params["window_size"]
        self.timestamp_start_of_day = 0
        self.processed = {"window_price": deque(maxlen=self.window_size),
                          "window_time": deque(maxlen=self.window_size)}
        self.norm_window = deque(maxlen=norm_window_size*60)
        self.avg = -1
        self.st_dev = 0

    def add_data(self, data):
        self.norm_window.append(data)

    def normalize_window(self):
        len_norm_window = len(self.norm_window)
        if len_norm_window < 3:
            self.avg = self.processed["window_price"][-1]
            self.st_dev = 0
        else:
            self.st_dev = stdev(self.norm_window)
            self.avg = np.mean(self.norm_window)

        norm_values = []
        for i in range(len(self.processed["window_price"])):
            if len_norm_window < 3:
                norm_values.append(0)
            else:
                norm_values.append((self.processed["window_price"][i] - self.avg)/(self.st_dev+0.000001))
        return norm_values

    def get_window(self):
        norm_prices = self.normalize_window()
        return norm_prices, self.processed["window_time"], self.avg, self.st_dev

    def clear(self):
        self.processed["window_price"].clear()
        self.processed["window_time"].clear()
        self.norm_window.clear()

    def is_window_filled(self):
        return len(self.processed["window_price"]) == self.window_size

    def process(self, market_order):
        time = market_order["publication_time"]
        price = market_order["price"]
        self.processed["window_price"].append(price)
        if len(self.processed["window_time"]) == 0:
            dt = datetime.fromtimestamp(time)
            stamp = datetime(dt.year, dt.month, dt.day, 9, 0).timestamp()
            self.timestamp_start_of_day = int(stamp)
        self.processed["window_time"].append((time - self.timestamp_start_of_day)/30600) # 8.5*60*60

    def gen_rows(self, df):
        for row in df.itertuples(index=False):
            yield row._asdict()

    def process_start_window(self, df):
        start_time = df["publication_time"].iloc[0]
        next_day = datetime.fromtimestamp(start_time).date() + timedelta(days=1)
        next_day_timestamp = datetime.combine(next_day, datetime.min.time()).timestamp()
        data = df[(df["publication_time"] < next_day_timestamp)]
        market_orders = self.gen_rows(data)
        for market_order in market_orders:
            self.process(market_order)
        return next_day_timestamp
