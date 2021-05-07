import math
import time

import numpy

from input_handler import InputHandler
from threading import Thread


class FileInput:

    def __init__(self, file, input_handler: InputHandler, input_size, reads_per_second=1, benchmark=False):
        self.reads_per_second = reads_per_second
        self.file = file
        self.input_handler = input_handler
        self.input_size = input_size
        self.benchmark = benchmark

    def start(self):
        thread = Thread(target=self.__start_reading())
        thread.start()

    def __start_reading(self):
        csv_file = open(self.file, 'r')
        next(csv_file)  # Skip CSV headers'
        start_time = -1
        while True:

            line = csv_file.readline()

            if not line:  # if line is empty end of file is reached
                break
            arr = line.split(";")
            arr = [float(i) for i in arr]
            #print(str(time.time()) + " : " + str(arr[-1]))
            time.sleep(math.ceil(time.time())-time.time())

            if self.benchmark:
                start_time = time.time()
            self.input_handler.put_all(arr[-1], numpy.array(arr[:self.input_size]), start_time=start_time)


