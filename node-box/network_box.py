import logging
import threading
from threading import Thread
import socket
import json
from observer import Observer
from input_handler import InputHandler


class InputThread(Thread):
    def __init__(self, connection, input_handler: InputHandler, size=1024):
        Thread.__init__(self)
        self.c = connection
        self.size = size
        self.input_handler = input_handler

    def run(self):
        while True:
            raw_data = self.c.recv(self.size)
            if len(raw_data) == 0:
                print("Connection lost, closing thread")
                self.c.close()
                break
            data = json.loads(raw_data.decode("utf-8"))
            self.input_handler.put(int(data["ts"]), data["data"], data["tag"])


class NetworkInput:
    def __init__(self, input_handler: InputHandler):
        self.input_handler = input_handler

    def connect(self, ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(30)
        sock.connect((ip, port))
        sock.settimeout(None)
        input_thread = InputThread(sock, self.input_handler)
        input_thread.start()


class NetworkOutput(Observer):
    def __init__(self, port, _id, tags: list, logger: logging):
        self.logger = logger
        self.tags = tags
        self.id = _id
        self.connections = []
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('', port))
        self.server_socket.listen(99)
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.start()

    def _run_server(self):
        while True:
            connection, (ip, port) = self.server_socket.accept()
            self.logger.info(str(self.id) + ' got connection from (' + str(ip) + ',' + str(port) + ')')
            self.connections.append(connection)

    def notify(self, result: (int, list)):
        data = {"id": str(self.id),
                "ts": str(result[0]),
                "tag": self.tags,
                "data": result[1]}
        self.logger.info(data)
        for c in self.connections:
            c.send(json.dumps(data).encode("utf-8"))
