import threading
from threading import Thread
import socket
import json
from torch import float32
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
            print(data)
            self.input_handler.put(int(data["ts"]), int(data["id"]), float(data["data"]))


class NetworkInput:
    def __init__(self, input_handler: InputHandler):
        self.input_handler = input_handler

    def connect(self, ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect((ip, port))
        sock.settimeout(None)
        input_thread = InputThread(sock, self.input_handler)
        input_thread.start()


class NetworkOutput(Observer):
    def __init__(self, port, _id):
        self.id = _id
        self.connections = []
        self.server_socket = socket.socket()
        self.server_socket.bind(('', port))
        self.server_socket.listen(5)
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.start()

    def _run_server(self):
        while True:
            connection, (ip, port) = self.server_socket.accept()
            print(str(self.id) + ' got connection from ', (ip, port))
            self.connections.append(connection)

    def notify(self, result: (int, float32)):
        for c in self.connections:
            data = {"id": str(self.id),
                    "ts": str(result[0]),
                    "data": str(result[1].item())}
            c.send(json.dumps(data).encode("utf-8"))
