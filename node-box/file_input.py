import time

import numpy

from input_handler import InputHandler
from threading import Thread


class FileInput:

    def __init__(self, file, input_handler: InputHandler, input_size, reads_per_second=10):
        self.reads_per_second = reads_per_second
        self.file = file
        self.input_handler = input_handler
        self.input_size = input_size

    def start(self):
        thread = Thread(target=self.__start_reading())
        thread.start()

    def __start_reading(self):
        csv_file = open(self.file, 'r')
        next(csv_file)  # Skip CSV headers
        while True:
            time.sleep(1/self.reads_per_second)
            line = csv_file.readline()

            if not line:  # if line is empty end of file is reached
                break
            arr = line.split(";")
            arr = [float(i) for i in arr]
            self.input_handler.put_all(arr[-1], numpy.array(arr[:self.input_size]))


