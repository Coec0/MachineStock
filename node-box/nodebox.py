from network_box import *
from node_box_processor import NodeBoxProcessor
from numpy import ndarray
from torch import float32
import json


class ExampleProcessor(NodeBoxProcessor):
    def __init__(self):
        self.ts = 0

    def process(self, features: ndarray) -> (int, float32):
        self.ts += 1
        return self.ts, self.ts*self.ts


class NodeBox:
    def __init__(self, coord_ip, coord_port, layer, input_size, ws=10):
        self.config = self.__fetch_coordinator_config(coord_ip, coord_port, layer)
        print(self.config)
        output_network = NetworkOutput(self.config["port"], self.config["id"])
        processor = ExampleProcessor()
        input_handler = InputHandler(ws, input_size, processor, output_network)
        self.network_input = NetworkInput(input_handler)
        self.connect()

    def connect(self):
        for ip, port in self.config["server_ip_port"]:
            self.network_input.connect(ip, port)

    @staticmethod
    def __fetch_coordinator_config(coord_ip, coord_port, layer):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((coord_ip, coord_port))
        sock.send(json.dumps({"layer": layer}).encode("utf-8"))
        return json.loads(sock.recv(1024).decode("utf-8"))

