import torch
import numpy as np
import collections
from adapter import Adapter
import threading
from data_handler import DataHandler

class StockPredictor:

    def __init__(self, stocks, input_buffer_size):
        self.stocks = stocks
        self.input_buffer_size = input_buffer_size

        host = "127.0.0.1"
        port = 2000
        print("ADAPTER STARTING")
        self.input_adapter = Adapter(host, port, stocks)
        print("ADAPTER ACCEPTED")

        parameters = {
            "build_delay" : 1,
            "stocks" : stocks,
            "math_features" : ["average", "variance"],
            "nbr_market_orders" : input_buffer_size,
            "market_order_features" : ["price", "volume", "mmt_flags"]}

        self.data_handler = DataHandler(self.input_adapter, parameters)
        print("DATAHANDLER ACCEPTED")
        self.load_model()

    def load_model(self, suffix=""):
        suffix = "-"+suffix if suffix != "" else ""
        self.model = torch.load("models/"+self.stocks[0]+suffix+".model") #TEmp only one stock


    def send_prediction(pred):
        print("Prediction: "+str(pred))

    def inference(self):
        x = torch.tensor(self.market_order_buffer)
        y = self.model.eval(x)
        return y

    def start(self):
        while(True):
            input = self.data_handler.get_input_data() #Prob need to wait if no data available
            if(len(self.market_order_buffer) == self.input_buffer_size):
                y = self.inference()
                self.send_prediction(y)
