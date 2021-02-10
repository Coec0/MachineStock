import socket
import struct
import threading
import queue
import json

class Adapter:

    def __init__(self, host='127.0.0.1', port=2000):
        self.host = host
        self.port = port
        self.queue = queue.Queue()
        threading.Thread(target=self.__tcp, daemon=True).start() #Start listening
    
    def __tcp(self): 
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            while(True):
                msg_len_raw = s.recv(4) #Read header to get size of message
                msg_len = struct.unpack('>I', msg_len_raw)[0]
                self.queue.put(s.recv(msg_len).decode())

    #If a timeout is set and a timeout is reached, an "Empty" exception will be raised
    def get(self, time_out=None):
        return json.loads(self.queue.get(timeout=time_out))
