from collections import deque
import torch
import numpy as np
from numpy import ndarray
from torch import nn
import torch.nn.functional as f

from processors.node_box_processor import NodeBoxProcessor


class DeepNetworkProcessor(NodeBoxProcessor):
    def __init__(self, weights_file, input_size, time: bool):
        self.time = time
        self.queue_price = deque(maxlen=input_size)
        self.queue_time = deque(maxlen=input_size)
        self.previous_timestamp = None
        self.model = DeepModel(input_size)
        if weights_file is not None:
            self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()

    def process(self, timestamp, features: ndarray) -> (int, list):
        """Process the data and return the result as a tuple of (timestamp, result).
        The timestamp is the timestamp of when the result is predicted for """
        if self.previous_timestamp is None or timestamp - self.previous_timestamp > 60*60*10:
            for _ in range(self.queue_price.maxlen):
                self.queue_price.append(features[0])
                if self.time:
                    self.queue_time.append(features[1])
        else:
            self.queue_price.append(features[0])
            if self.time:
                self.queue_time.append(features[1])

        self.previous_timestamp = timestamp

        ml_array = np.array(self.queue_price)
        if self.time:
            ml_array += np.array(self.queue_time)

        return int(timestamp), [self.predict(ml_array).item()]

    def predict(self, features: ndarray):
        with torch.no_grad():
            x = torch.tensor(features).type(torch.FloatTensor)
            return self.model(x)


class DeepModel(nn.Module):
    def __init__(self, input_size):
        data_type = torch.FloatTensor
        #device = torch.device('cuda:0')
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size*2).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2 = nn.Linear(input_size*2, round(input_size*1.5)).type(data_type)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3 = nn.Linear(round(input_size*1.5), round(input_size*0.5)).type(data_type)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4 = nn.Linear(round(input_size*0.5), 20).type(data_type)
        self.fc4.weight.data.uniform_(-0.1, 0.1)
        self.fc5 = nn.Linear(20, 1).type(data_type)
        self.fc5.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))
        x = f.leaky_relu(self.fc3(x))
        x = f.leaky_relu(self.fc4(x))
        x = f.leaky_relu(self.fc5(x))
        return x
