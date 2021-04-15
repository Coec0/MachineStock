from threading import Thread
from smartsync.smart_sync import SmartSync
from node_box_processor import NodeBoxProcessor
from observer import Observer


class NodeBoxWrapper:
    def __init__(self, ws, number_of_features, processor: NodeBoxProcessor, observer: Observer):
        self.number_of_features = number_of_features
        self.smart_sync = SmartSync(ws, number_of_features)
        self.processor = processor
        self.observer = observer

    def put(self, timestamp, node_number, value):
        arr = self.smart_sync.put(timestamp, node_number, value)
        if arr is not None:
            thread = Thread(target=self.__process_arr, args=(arr,))
            thread.start()

    def __process_arr(self, arr):
        self.observer.notify(self.processor.process(arr))

