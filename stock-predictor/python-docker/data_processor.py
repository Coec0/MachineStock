import queue
from collections import deque
from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer

class DataProcessor:
    def __init__(self, stock, window_size, usePrice=True, beta=0.98, useVol=False ,useExpAvgPrice=True): #["price, volume, --mnt--p, publicationtime"]
        self.useExpAvgPrice = useExpAvgPrice
        self.usePrice = usePrice
        self.useVol = useVol
        self.stock = stock
        self.beta = beta
        self.window_size = window_size
        self.processed = {"window" : deque(maxlen=window_size),
                          "exp_avg_price" : 0}

    def gen_rows(self, df):
        for row in df.itertuples(index=False):
            yield row._asdict()

    def get_window(self):
        return self.processed["window"]

    def get_financial_models(self):
        data = []
        if(self.useExpAvgPrice):
            data.append(self.processed["exp_avg_price"])
        return data

    def update_exp_avg_price(self, price):
        old_exp_avg = self.processed["exp_avg_price"]
        new_exp_avg = self.beta*old_exp_avg + (1-self.beta)*price
        self.processed["exp_avg_price"] = new_exp_avg

    def trim_market_order(self, market_order):
        mr = []
        if(self.usePrice):
            mr.append(market_order["price"])
        if(self.useVol):
            mr.append(market_order["volume"])
        return mr

    def clear(self):
        self.processed["window"].clear()

    def is_window_filled(self):
        return len(self.processed["window"]) == self.window_size

    def process(self, market_order): #JSON/dict
        mr = self.trim_market_order(market_order)
        self.processed["window"].append(mr)

        if(self.useExpAvgPrice):
            self.update_exp_avg_price(market_order["price"])

    # df - pandas dataframe sorted by publication_time
    def process_start_window(self, df):
        start_time = df["publication_time"].iloc[0]
        next_day = datetime.fromtimestamp(start_time).date() + timedelta(days=1)
        next_day_timestamp = datetime.combine(next_day, datetime.min.time()).timestamp()
        data = df[(df["stock"]==self.stock) & (df["publication_time"] < next_day_timestamp)]
        market_orders = self.gen_rows(data)
        for market_order in market_orders:
            self.process(market_order)
        return next_day_timestamp
