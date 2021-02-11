import collections
import numpy as np
import time
import queue

class DataHandler:

    def __init__(self, input_adapter, parameters): #build_delay, input_adapter, stocks, math_features, nbr_market_orders):
        #  If time delay is zero then give next input vector when
        #  a market order is found

        #Paramters include: build_delay, stocks, math_features, market_order_paramters
        self.useAvg = False
        self.useVar = False

        self.build_delay = parameters["build_delay"]
        self.last_build_time = round(time.time())
        self.input_vector_queue = queue.Queue()
        self.input_adapter = input_adapter
        self.stocks = parameters["stocks"]
        self.set_math_features(parameters["math_features"])

        self.market_orders = {}
        self.price_sum = {}
        self.square_price_sum = {}
        self.stock_count = {}

        for stock in self.stocks:
            self.market_orders[stock] = collections.deque(maxlen=parameters["nbr_market_orders"])
            self.price_sum[stock] = 0
            self.square_price_sum[stock] = 0
            self.stock_count[stock] = 0

        self.run()

    def set_math_features(self,math_features):
        for t in math_features:
            if(t == "average"):
                self.useAvg = True
            elif(t == "variance"):
                self.useVar = True

    def process_data(self, market_order):
        stock = market_order.stock
        price = market_order.price
        #Onnly save relevant data
        self.market_orders[stock].extend(market_order)

        if(self.useAvg or self.useVar):
            self.stock_count[stock] += 1
        if(self.useAvg):
            self.price_sum[stock] += price
        if(self.useVar):
            self.square_price_sum[stock] += price*price

    def get_price_averages(self):
        averages = []
        for stock in self.stocks:
            avg = self.price_sum[stock] / self.stock_count[stock]
            averages.append(avg)
        return averages

    def get_price_variance(self):
        variances = []
        for stock in self.stocks:
            square_sum = self.square_price_sum[stock]
            n = self.stock_count[stock]
            square_avg = square_sum / n
            var = square_sum - square_avg / (n-1)
            variances.append(var)
        return variances

    def build_input(self):
        stack = []
        for stock in self.stocks:
            market_orders = np.array(self.market_orders[stock])
            stack.append(market_orders)

        if(self.useAvg):
            averages = self.get_price_averages()
            stack.append(averages)

        if(self.useVar):
            variances = self.get_price_variance()
            stack.append(variances)

        return np.vstack(stack)

    def run(self):
        while(True):
            market_order = self.input_adapter.get()
            self.process_data(market_order)
            now = round(time.time())
            if(now - self.last_build_time >= self.build_delay):
                input_vector = self.build_input()
                self.input_vector_queue.put(input_vector)
                self.last_build_time = round(time.time())

    def get_input_data(self):
        return self.input_vector_queue.get()
