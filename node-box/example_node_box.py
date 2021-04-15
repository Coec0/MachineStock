from numpy import ndarray
from torch import float32
from node_box_wrapper import NodeBoxWrapper
from node_box_processor import NodeBoxProcessor
from observer import Observer


class ExampleNodeBox(Observer):
    def __init__(self):
        self.ts = 0

    def notify(self, result: (int, float32)):
        print(result)


class ExampleProcessor(NodeBoxProcessor):
    def __init__(self):
        self.ts = 0

    def process(self, features: ndarray) -> (int, float32):
        self.ts += 1
        return self.ts, self.ts*self.ts


example_node_box = ExampleNodeBox()
processor = ExampleProcessor()
node_box_wrapper = NodeBoxWrapper(2, 3, processor, example_node_box)

for i in range(5):
    node_box_wrapper.put(i, 0, 1)
    node_box_wrapper.put(i, 1, 1)
    node_box_wrapper.put(i, 2, 1)
