from threading import Thread

from numpy import ndarray

from smartsync.smart_sync import SmartSync
from processors.node_box_processor import NodeBoxProcessor
from observer import Observer


class InputHandler:
    def __init__(self, ws, input_size, processor: NodeBoxProcessor, observer: Observer):
        self.number_of_features = input_size
        self.smart_sync = SmartSync(ws, input_size)
        self.processor = processor
        self.observer = observer

    def put(self, timestamp, node_number, values: list):
        arr = None
        for value in values:
            arr = self.smart_sync.put(timestamp, node_number, value)
        if arr is not None:
            thread = Thread(target=self.__process_arr, args=(timestamp, arr))
            thread.start()

    def put_all(self, timestamp, values: ndarray):
        thread = Thread(target=self.__process_arr, args=(timestamp, values))
        thread.start()

    def __process_arr(self, timestamp, arr):
        self.observer.notify(self.processor.process(int(timestamp), arr))

