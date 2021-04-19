from numpy import ndarray
from torch import float32
from coordinator import Coordinator
from coordinator_strategies.fully_connected_strategy import FullyConnectedStrategy
from nodebox import NodeBox
import threading


def start_node_box(layer):
    NodeBox("localhost", 5500, layer, 3)


Coordinator(5501, 4, FullyConnectedStrategy())
t1 = threading.Thread(target=start_node_box, args=(0,))
t1.start()
t2 = threading.Thread(target=start_node_box, args=(0,))
t2.start()
t3 = threading.Thread(target=start_node_box, args=(0,))
t3.start()
start_node_box(1)
#l1_0.connect("localhost", 12348)
#l1_1 = NodeBox(12346, "1", 1)

#l2 = NodeBox(12347, "2", 2)