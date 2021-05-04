from network_box import *
from file_input import FileInput
import json
import logging


class NodeBox:
    def __init__(self, coord_ip, coord_port, layer, input_size, processor, tags: list, tag_to_pos=None,
                 local_file=None, ws=10, verbosity=logging.WARNING, benchmark=False):
        self.config = self.__fetch_coordinator_config(coord_ip, coord_port, layer)
        self.local_file = local_file
        self.tags = tags
        logging.basicConfig(level=logging.NOTSET)
        logger = logging.getLogger(str(tags))
        logger.setLevel(verbosity)
        print(str(logger))
        logger.info(self.config)
        output_network = NetworkOutput(self.config["port"], self.config["id"], tags, logger)
        input_handler = InputHandler(ws, input_size, tag_to_pos, processor, output_network, logger)
        self.network_input = NetworkInput(input_handler)
        if local_file is not None:
            logger.info("Found local file " + local_file)
            self.local_input = FileInput(local_file, input_handler, input_size, benchmark=benchmark)
        self.connect()

    def connect(self):
        logger = logging.getLogger(str(self.tags))
        logger.info(str(self.tags) + "Connecting to " + str(len(self.config["server_ip_port"])) + " servers")

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
        data = sock.recv(65536).decode("utf-8")
        return json.loads(data)

