import queue
from collections import deque
from datetime import datetime
from datetime import timedelta
from timeit import default_timer as timer
from price_channels import PriceChannels
import math

class DataProcessor:
    def __init__(self, stock, params): #["price, volume, --mnt--p, publicationtime"]
        self.useChannels = "channels" in params["financial_models"]
        self.useEMA = "ema" in params["financial_models"]
        self.useRSI = "rsi" in params["financial_models"]
        self.useMACD = "macd" in params["financial_models"]
        self.useVolatility = "volatility" in params["financial_models"]
        self.usePrice = "price" in params["market_order_features"]
        self.useVol = "volume" in params["market_order_features"]
        self.stock = stock
        self.params = params
        self.window_size = params["window_size"]
        self.channels = PriceChannels(120, 10, params["normalize"])
        self.rsi = {"window" : deque(maxlen=14),
                    "open": 0,
                    "latest": 0,
                    "seg_start": -1}
        self.ema = {"window26" : deque(maxlen=26),
                    "beta": 0.98,
                    "seg_start": -1}
        self.volatility = {"window" : deque(maxlen=50),
                           "window_size":50,
                           "mean" : 0,
                           "varsum":0}
        self.processed = {"window" : deque(maxlen=self.window_size),
                          "rsi" : 0,
                          "ema12":0,
                          "ema26":0,
                          "macd" :0,
                          "volatility": 0}

    def gen_rows(self, df):
        for row in df.itertuples(index=False):
            yield row._asdict()

    def get_window(self):
        return self.processed["window"]

    def get_financial_models(self):
        data = []
        if(self.useEMA):
            data.append('%.4f' % self.processed["ema12"])
            data.append('%.4f' % self.processed["ema26"])
        if(self.useRSI):
            data.append('%.4f' % self.processed["rsi"])
        if(self.useMACD):
            data.append('%.4f' % self.processed["macd"])
        if(self.useVolatility):
            data.append('%.6f' % self.processed["volatility"])
        if(self.useChannels):
            data.append('%.4f' % self.channels.get_min_max_k())
            data.append('%.4f' % self.get_relativity_in_price_channel())
        return data

    def update_volatility(self, mo):
        new_x = mo["price"]
        window_size = self.volatility["window_size"]
        varsum = self.volatility["varsum"]

        if(len(self.volatility["window"])==0):
            old_x = mo["price"]+0.00001
            old_mean = old_x
        else:
            old_x = self.volatility["window"][0]
            old_mean = self.volatility["mean"]

        new_mean = old_mean + ((new_x - old_x)/window_size)
        #varsum = varsum + (new_x-old_mean)*(new_x-new_mean)-((old_x-old_mean)*(old_x-new_mean))
        varsum = varsum + (new_x + old_x - old_mean - new_mean) * (new_x-old_x)
        varsum = max(0, varsum)

        self.volatility["window"].append(new_x)
        self.processed["volatility"] = math.sqrt(varsum/(window_size-1))
        self.volatility["mean"] = new_mean
        self.volatility["varsum"] = varsum

    def update_ema(self, mo):
        if(self.ema["seg_start"] == -1):
            self.ema["seg_start"] = mo["publication_time"]
        if(mo["publication_time"] > self.ema["seg_start"]+5*60):
            self.ema["window26"].append(mo["price"])
            ema12, ema26 = self.calc_ema()
            self.processed["ema12"] = ema12
            self.processed["ema26"] = ema26

    def calc_ema(self):
        count = 0
        ema12 = self.processed["ema12"]
        ema26 = self.processed["ema26"]
        w12 = 2 / 13
        w26 = 2 / 27
        for p in self.ema["window26"]:
            ema26 = w26*p + (1-w26)*ema26
            if(count >= 14):
                ema12 = w12*p + (1-w12)*ema12
            count += 1
        return (ema12,ema26)

    def update_rsi(self, mo):
        if(self.rsi["seg_start"] == -1):
            self.rsi["seg_start"] = mo["publication_time"]
        if(mo["publication_time"] > self.rsi["seg_start"]+5*60):
            diff = self.rsi["open"] - self.rsi["latest"]
            while(mo["publication_time"] > self.rsi["seg_start"]+5*60):
                self.rsi["window"].append(diff)
                self.rsi["seg_start"] = self.rsi["seg_start"]+5*60
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

        #reset ema
        self.ema["seg_start"] = -1
        #self.ema["window26"].clear()

    def is_window_filled(self):
        return len(self.processed["window"]) == self.window_size

    def process(self, market_order): #JSON/dict
        mr = self.trim_market_order(market_order)
        self.processed["window"].append(mr)

        if(self.useEMA or self.useMACD):
            self.update_ema(market_order)

        if(self.useRSI):
            self.update_rsi(market_order)

        if(self.useMACD):
            self.processed["macd"] = self.processed["ema12"] - self.processed["ema26"]

        if(self.useVolatility):
            self.update_volatility(market_order)

        if(self.useChannels):
            self.channels.update(market_order)

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
