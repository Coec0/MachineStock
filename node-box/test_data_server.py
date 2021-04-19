import socket
import json
import threading
import sys


port = sys.argv[1]
_id = sys.argv[2]
host = "localhost"

ts = 0
data = {"id": _id,
        "ts": ts,
        "data": ""}

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, int(port)))
server_socket.listen(5)
connections = []


def _server():
    while True:
        connection, (ip, _port) = server_socket.accept()
        connections.append(connection)
        print('Got connection from ', (ip, _port))


server_thread = threading.Thread(target=_server)
server_thread.start()

while True:
    msg = input("message: ")
    data["data"] = msg
    jsonStr = json.dumps(data)
    print(jsonStr)
    for c in connections:
        c.send(jsonStr.encode("utf-8"))
