import queue
from collections import deque
from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer

class DataProcessor:
<<<<<<< HEAD
    def __init__(self, stock, params, beta=0.98): #["price, volume, --mnt--p, publicationtime"]
        self.useExpAvgPrice = "exp_avg_price" in params["financial_models"]
        self.useRSI = "rsi" in params["financial_models"]
        self.usePrice = "price" in params["market_order_features"]
        self.useVol = "volume" in params["market_order_features"]
=======
    def __init__(self, stock, window_size, usePrice=True, beta=0.98, useVol=False ,useExpAvgPrice=True): #["price, volume, --mnt--p, publicationtime"]
        self.useExpAvgPrice = useExpAvgPrice
        self.usePrice = usePrice
        self.useVol = useVol
>>>>>>> 743f6446195f1dee768be844f5f2fe26ec59fc47
        self.stock = stock
        self.beta = beta
        self.params = params
        self.window_size = params["window_size"]
        self.rsi = {"window" : deque(maxlen=14),
                    "open": 0,
                    "latest": 0,
                    "seg_start": -1}
        self.processed = {"window" : deque(maxlen=self.window_size),
                          "exp_avg_price" : 0,
                          "rsi" : 0}

    def gen_rows(self, df):
        for row in df.itertuples(index=False):
            yield row._asdict()

    def get_window(self):
        return self.processed["window"]

    def get_financial_models(self):
        data = []
        if(self.useExpAvgPrice):
            data.append(self.processed["exp_avg_price"])
        if(self.useRSI):
            data.append(self.processed["rsi"])
        return data

    def update_exp_avg_price(self, price):
        old_exp_avg = self.processed["exp_avg_price"]
        new_exp_avg = self.beta*old_exp_avg + (1-self.beta)*price
        self.processed["exp_avg_price"] = new_exp_avg

    def update_rsi(self, mo):
        if(self.rsi["seg_start"] == -1):
            self.rsi["seg_start"] = mo["publication_time"]
        if(mo["publication_time"] > self.rsi["seg_start"]+5*60):
            diff = self.rsi["open"] - self.rsi["latest"]
            self.rsi["window"].append(diff)
            self.rsi["seg_start"] = self.rsi["seg_start"]+5*60 #must fix later
            self.rsi["open"] = mo["price"]
            self.processed["rsi"] = self.calc_rsi()
        self.rsi["latest"] = mo["price"]

    def calc_rsi(self):
        sumUP = 0
        sumDW = 0.000001
        for x in self.rsi["window"]:
            if x > 0:
                sumUP += x
            else:
                sumDW += x
        rs = sumUP / abs(sumDW)
        rsi = 100 - (100/(1+rs))
        return rsi

    def trim_market_order(self, market_order):
        mr = []
        if(self.usePrice):
            mr.append(market_order["price"])
        if(self.useVol):
            mr.append(market_order["volume"])
        return mr

    def clear(self):
        self.processed["window"].clear()

        #reset rsi
        self.rsi["seg_start"] = -1
        self.rsi["window"].clear()

    def is_window_filled(self):
        return len(self.processed["window"]) == self.window_size

    def process(self, market_order): #JSON/dict
        mr = self.trim_market_order(market_order)
        self.processed["window"].append(mr)

        if(self.useExpAvgPrice):
            self.update_exp_avg_price(market_order["price"])

        if(self.useRSI):
            self.update_rsi(market_order)

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
