import math

import torch
from numpy import ndarray
from torch import nn
import torch.nn.functional as f
from processors.node_box_processor import NodeBoxProcessor


class CombinerProcessor(NodeBoxProcessor):
    def __init__(self, weights_file, input_size):
        self.model = CombinerModel(input_size)
        if weights_file is not None:
            self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()

    def process(self, timestamp, features: ndarray) -> (int, list):
        """Process the data and return the result as a tuple of (timestamp, result).
        The timestamp is the timestamp of when the result is predicted for

        features: price, ema, rsi, macd, volatility, channels"""
        return int(timestamp), [self.predict(features).item()]

    def predict(self, features: ndarray):
        with torch.no_grad():
            x = torch.tensor(features).type(torch.FloatTensor)
            return self.model(x)


class CombinerModel(nn.Module):
    def __init__(self, input_size):
        data_type = torch.FloatTensor
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size*2).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2 = nn.Linear(input_size*2, math.ceil(input_size*0.5)).type(data_type)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3 = nn.Linear(math.ceil(input_size*0.5), 1).type(data_type)
        self.fc3.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))
        y = f.leaky_relu(self.fc3(x))
        return y
