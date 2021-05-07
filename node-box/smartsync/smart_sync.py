import logging
from multiprocessing import shared_memory, Lock
import numpy
import numpy as np


class SmartSync:
    def __init__(self, ws, number_of_nodes, logger=None, mem_name_main=None, mem_name_non_nan=None):
        self.logger = logger
        self.mem_name_main = mem_name_main
        self.mem_name_non_nan = mem_name_non_nan
        self.ws = ws
        self.number_of_nodes = number_of_nodes
        if mem_name_main is None:
            self.collector1 = np.full([ws, number_of_nodes + 1], np.NaN, dtype=np.float32)  # First position represents the int_div of ws
            self.collector1[:, 0] = 0
            self.non_nan_in_row = np.zeros(ws, dtype=np.int32)
        #else:
            #self.init_shared_memory()

        """self.collector1 = np.full([ws, number_of_nodes + 1], np.NaN)  # First position represents the int_div of ws
        self.collector1[:, 0] = 0
        self.non_nan_in_row = np.zeros(ws)

        shm = shared_memory.SharedMemory(create=True, size=self.collector1.nbytes)
        collector = np.ndarray(self.collector1.shape, dtype=self.collector1.dtype, buffer=shm.buf)
        collector[:] = self.collector1[:]  # Copy the original data into shared memory
        self.collector1 = collector

        shm = shared_memory.SharedMemory(create=True, size=self.non_nan_in_row.nbytes)
        non_nan = np.ndarray(self.non_nan_in_row.shape, dtype=self.non_nan_in_row.dtype, buffer=shm.buf)
        non_nan[:] = self.non_nan_in_row[:]  # Copy the original data into shared memory
        self.non_nan_in_row = non_nan"""

    def get_shared_memory(self):
        existing_shm = shared_memory.SharedMemory(name=self.mem_name_main)
        collector = np.ndarray((self.ws, self.number_of_nodes + 1), dtype=np.float32, buffer=existing_shm.buf)
        existing_shm2 = shared_memory.SharedMemory(name=self.mem_name_non_nan)
        non_nan_in_row = np.ndarray((self.ws,), dtype=np.int32, buffer=existing_shm2.buf)
        print(collector)
        return collector, non_nan_in_row

    #  node_number should start at 0
    #  Returns all values as an numpy array for the current timestamp if it is full, else None
    def put(self, timestamp, node_number, value):
        if self.mem_name_main is not None:
            existing_shm = shared_memory.SharedMemory(name=self.mem_name_main)
            collector = np.ndarray((self.ws, self.number_of_nodes + 1), dtype=np.float, buffer=existing_shm.buf)
            existing_shm2 = shared_memory.SharedMemory(name=self.mem_name_non_nan)
            non_nan_in_row = np.ndarray((self.ws,), dtype=np.int64, buffer=existing_shm2.buf)
        else:
            collector = self.collector1
            non_nan_in_row = self.non_nan_in_row

        node_number = node_number % self.number_of_nodes
        arr_pos = timestamp % self.ws
        int_div = timestamp // self.ws

        #if self.mem_name_main is not None:
        #    self.lock.acquire()

        if collector[arr_pos][0] < int_div:  # Clear row if cycled through array
            collector[arr_pos][0] = int_div
            non_nan_in_row[arr_pos] = 0
            if self.logger is not None:
                self.logger.debug("Smartsync clearing row " + str(arr_pos))
            for i in range(self.number_of_nodes):
                collector[arr_pos][i + 1] = np.NaN
        elif collector[arr_pos][0] > int_div:  # If int_div is from previous cycle
            print("Returning None")
            return None

        collector[arr_pos][node_number + 1] = value
        non_nan_in_row[arr_pos] += 1
       # if self.mem_name_main is not None:
        #    self.lock.release()

        if self.logger is not None:
            self.logger.debug("Smartsync value added to [" + str(arr_pos) + "][" + str(node_number + 1) + "]")
            self.logger.debug(
                str(np.count_nonzero(~np.isnan(collector[arr_pos][1:]))) + " of " + str(self.number_of_nodes)
                + " values are filled in the row")

        # if self.__collector_row_filled(self.collector1[arr_pos][1:]):
        if non_nan_in_row[arr_pos] == self.number_of_nodes:
            if self.logger is not None:
                self.logger.info("Row " + str(arr_pos) + " filled, returning values")
            return np.array(collector[arr_pos][1:])
        return None

    @staticmethod
    def __collector_row_filled(row):
        return not numpy.isnan(row).any()


    @staticmethod
    def get_shared_memory_names(ws, number_of_nodes):
        collector = np.full([ws, number_of_nodes + 1], np.NaN, dtype=np.float)
        collector[:, 0] = 0
        non_nan_in_row = np.zeros(ws, dtype=np.int64)

        shm_collector = shared_memory.SharedMemory(create=True, size=collector.nbytes)
        collector_shm = np.ndarray(collector.shape, dtype=collector.dtype, buffer=shm_collector.buf)
        collector_shm[:] = collector[:]  # Copy the original data into shared memory

        shm_nan_in_row = shared_memory.SharedMemory(create=True, size=non_nan_in_row.nbytes)
        nan_in_row_shm = np.ndarray(non_nan_in_row.shape, dtype=non_nan_in_row.dtype, buffer=shm_nan_in_row.buf)
        nan_in_row_shm[:] = non_nan_in_row[:]  # Copy the original data into shared memory

        return shm_collector.name, shm_nan_in_row.name
