import torch
from torch import nn
from time import sleep
import threading
from threading import Thread
import socket
import json
import traceback


class InputThread(Thread):
    def __init__(self, ip, port, connection, inputs, inputs_ready, size=1024):
        Thread.__init__(self)
        self.c = connection
        self.port = port
        self.ip = ip
        self.size = size
        self.inputs = inputs
        self.inputs_ready = inputs_ready
        print("Start new client thread for IP:" + str(ip) + " PORT:" + str(port))

    # Tmp function to handle this
    def assign_data_list_tmp(self):
        idx = int(self.data["id"])
        self.inputs[idx] = float(self.data["data"])
        self.inputs_ready[idx] = True

    def run(self):
        while True:
            raw_data = self.c.recv(self.size)
            if len(raw_data) == 0:
                print("Connetion lost, closing thread")
                self.c.close()
                break
            self.data = json.loads(raw_data.decode("utf-8"))
            self.assign_data_list_tmp()
            print(self.inputs)

class Box:
    def __init__(self, net, id_, input_size, name=None, server_params=None):
        self.name = "Id" + str(id_) if name == None else name
        self.id = id_
        self.net = net
        self.input_size = input_size

        self.input_connections = []
        self.output_connections = []

        self.listeners = []

        self.inputs = [0] * input_size
        self.inputs_ready = [False] * input_size

        if server_params != None:
            self.server_socket = socket.socket()
            self.server_socket.bind(('', server_params["port"]))
            self.server_socket.listen(5)
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.start()

        self.pred_thread = threading.Thread(target=self._predict)
        self.pred_thread.start()

    def _run_server(self):
        while (True):
            connection, (ip, port) = self.server_socket.accept()
            print('Got connection from ', (ip, port))
            # TODO
            inputThread = InputThread(ip, port, connection, self.inputs, self.inputs_ready)
            inputThread.start()
            self.input_connections.append(inputThread)

    def create_output_connection(self, ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        self.output_connections.append(sock)

    # def set_input(self, idx, input_):
    #    self.inputs[idx] = input_
    #    self.inputs_ready[idx] = True

    def set_inputs(self, inputs):
        self.inputs = inputs
        self.inputs_ready = [True] * self.input_size

    def create_input_tensor(self):
        return torch.FloatTensor(self.inputs)

    def send_output(self, listener, output):
        data = {"id": str(self.id),
                "data": str(output)}
        jsonStr = json.dumps(data)
        listener.send(jsonStr.encode("utf-8"))

    def send_prediction(self, prediction):
        for listener in self.output_connections:
            self.send_output(listener, prediction)

    def _predict(self):
        while True:
            if all(self.inputs_ready):
                prediction = self.net(self.create_input_tensor()).item()
                self.send_prediction(prediction)
                print(self.name + "\tPred: " + str(prediction))
                self.inputs_ready = [False] * self.input_size
                break
