import torch
import numpy as np
import collections
import adapter
import threading
import data_handler

class StockPredictor:

    def __init__(self, stockname, input_buffer_size):
        self.stockname = stockname
        self.input_buffer_size = input_buffer_size
        self.input_adapter = Adapter()

        parameters =
        {
            "build_delay" : 1,
            "stocks" : ["Ericsson_A"],
            "math_features" : ["average", "variance"],
            "nbr_market_orders" : 20 
        }

        self.data_handler = DataHandler(self.input_adapter, parameters)
        self.load_model()
        self.start()

    def load_model(self, suffix=""):
        suffix = "-"+suffix if suffix != "" else ""
        self.model = torch.load("models/"+self.stockname+suffix+".model")

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
