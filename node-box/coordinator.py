import json
import time
import logging
import threading
import socket
from coordinator_strategies.strategy import Strategy


class Coordinator:
    def __init__(self, port, number_of_node_boxes, strategy: Strategy, verbosity=logging.WARNING,
                 sync_time_factor=0.02):
        logging.basicConfig(level=logging.NOTSET)
        self.logger = logging.getLogger(str("Coordinator"))
        self.logger.setLevel(verbosity)
        self.sync_time_factor = sync_time_factor
        self.number_of_node_boxes = number_of_node_boxes
        self.strategy = strategy
        self.start_port = 49152
        self.start_id = 0
        self.connections = {}
        self.layer_dict = {}
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('', port))
        self.server_socket.listen(number_of_node_boxes + 1)
        self.server_thread = threading.Thread(target=self.__run_server)
        self.server_thread.start()

    def __run_server(self):
        while len(self.connections) < self.number_of_node_boxes:
            connection, (ip, port) = self.server_socket.accept()
            self.logger.info('Coordinator got connection from (' + str(ip) + ',' + str(port) + ')')
            self.connections[(ip, port)] = connection
            node_info = json.loads(connection.recv(1024).decode("utf-8"))
            node_info["local_ip"] = ip
            node_info["local_port"] = port
            self.__configure_node_dict(node_info)
        self.__coordinate()

    def __configure_node_dict(self, node_info):
        node_info["port"] = self.__get_port()  # Port that should be used for node-box output layer
        node_info["id"] = self.start_id
        self.start_id += 1
        layer = int(node_info["layer"])
        if layer not in self.layer_dict.keys():
            self.layer_dict[layer] = []
        self.layer_dict[layer].append(node_info)

    def __get_port(self):
        for port in range(self.start_port, 65535):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    self.start_port = port + 1
                    return port
        raise ConnectionError("Couldn't find an open port")

    def __coordinate(self):
        self.strategy.execute(self.layer_dict)
        self.__return_strategy()

    def __return_strategy(self):
        start_time = time.time()
        sync_time = start_time + (self.sync_time_factor * self.number_of_node_boxes)
        for layer in self.layer_dict.values():
            for node in layer:
                node["sync_time"] = sync_time
                c = self.connections[(node["local_ip"], node["local_port"])]
                c.send(json.dumps(node).encode("utf-8"))
        print("Coordinator sent all configs in " + str(time.time() - start_time) + " seconds")
