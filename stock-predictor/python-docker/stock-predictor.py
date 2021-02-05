import torch
import numpy as np
import collections

class StockPredictor:

    def __init__(self, stockname, input_buffer_size):
        self.stockname = stockname
        self.input_buffer_size = input_buffer_size
        self.data_buffer = collections.deque(maxlen=input_buffer_size)
        self.load_model()
        self.set_input_adapter()
        self.start()

    def load_model(self, suffix=""):
        suffix = "-"+suffix if suffix != "" else ""
        self.model = torch.load("models/"+self.stockname+suffix+".model")

    def set_input_adapter(self):
        self.input_adapter = None
        print("Todo: Create input adapter")

    def fetch_data(self):
        data = self.input_adapter.fetch()
        self.data_buffer.extend(data)

    def send_prediction(pred):
        print("Prediction: "+str(pred))

    def inference(self):
        x = torch.tensor(self.data_buffer)
        y = self.model.eval(x)
        return y

    def start(self):
        while(True):
            self.fetch_data()
            if(len(self.data_buffer) == self.input_buffer_size):
                y = self.inference()
                self.send_prediction(y)
            sleep(1)
