from threading import Thread
from smartsync.smart_sync import SmartSync
from node_box_processor import NodeBoxProcessor
from observer import Observer


class InputHandler:
    def __init__(self, ws, input_size, processor: NodeBoxProcessor, observer: Observer):
        self.number_of_features = input_size
        self.smart_sync = SmartSync(ws, input_size)
        self.processor = processor
        self.observer = observer

    def put(self, timestamp, node_number, value):
        arr = self.smart_sync.put(timestamp, node_number, value)
        if arr is not None:
            thread = Thread(target=self.__process_arr, args=(arr,))
            thread.start()

    def __process_arr(self, arr):
        self.observer.notify(self.processor.process(arr))

