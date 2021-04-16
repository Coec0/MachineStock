from numpy import ndarray
from torch import float32

from nodebox import NodeBox



l1_0 = NodeBox(12345, "2", 2)
l1_0.connect("localhost", 12348)
#l1_1 = NodeBox(12346, "1", 1)

#l2 = NodeBox(12347, "2", 2)