from threading import Thread

from numpy import ndarray

from smartsync.smart_sync import SmartSync
from processors.node_box_processor import NodeBoxProcessor
from observer import Observer


class InputHandler:
    def __init__(self, ws, input_size, tag_to_pos: dict, processor: NodeBoxProcessor, observer: Observer):
        self.number_of_features = input_size
        self.tag_to_pos = tag_to_pos
        self.smart_sync = SmartSync(ws, input_size)
        self.processor = processor
        self.observer = observer

    def put(self, timestamp, values: list, tag):
        arr = None
        pos = self.tag_to_pos[tag]
        for value in values:
            arr = self.smart_sync.put(timestamp, pos, value)
        if arr is not None:
            thread = Thread(target=self.__process_arr, args=(timestamp, arr))
            thread.start()

    def put_all(self, timestamp, values: ndarray):
        thread = Thread(target=self.__process_arr, args=(timestamp, values))
        thread.start()

    def __process_arr(self, timestamp, arr):
        self.observer.notify(self.processor.process(int(timestamp), arr))

