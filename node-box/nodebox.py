from network_box import *
from node_box_processor import NodeBoxProcessor
from numpy import ndarray
from torch import float32


class ExampleProcessor(NodeBoxProcessor):
    def __init__(self):
        self.ts = 0

    def process(self, features: ndarray) -> (int, float32):
        self.ts += 1
        return self.ts, self.ts*self.ts

class NodeBox:
    def __init__(self, port, _id, input_size, ws=10):
        output_network = NetworkOutput(port, _id)
        processor = ExampleProcessor()
        input_handler = InputHandler(ws, input_size, processor, output_network)
        self.network_input = NetworkInput(input_handler)

    def connect(self, ip, port):
        self.network_input.connect(ip, port)
