import socket
import struct

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 2000        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while(True):
        msg_len_raw = s.recv(4) #Read header to get size of message
        msg_len = struct.unpack('>I', msg_len_raw)[0]
        print(s.recv(msg_len))
        print()

