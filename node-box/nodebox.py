from network_box import *
from node_box_processor import NodeBoxProcessor
from file_input import FileInput
from numpy import ndarray
from torch import float32
import json


class NodeBox:
    def __init__(self, coord_ip, coord_port, layer, input_size, processor, local_file=None, ws=10):
        self.config = self.__fetch_coordinator_config(coord_ip, coord_port, layer)
        self.local_file = local_file
        print(self.config)
        output_network = NetworkOutput(self.config["port"], self.config["id"])
        input_handler = InputHandler(ws, input_size, processor, output_network)
        self.network_input = NetworkInput(input_handler)
        if local_file is not None:
            self.local_input = FileInput(local_file, input_handler, input_size)
        self.connect()

    def connect(self):
        for ip, port in self.config["server_ip_port"]:
            self.network_input.connect(ip, port)

        # If no ip to connect to, try to read local file
        if len(self.config["server_ip_port"]) == 0 and self.local_file is not None:
            self.local_input.start()



    @staticmethod
    def __fetch_coordinator_config(coord_ip, coord_port, layer):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((coord_ip, coord_port))
        sock.send(json.dumps({"layer": layer}).encode("utf-8"))
        return json.loads(sock.recv(1024).decode("utf-8"))

