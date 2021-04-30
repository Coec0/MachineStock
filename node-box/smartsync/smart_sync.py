import logging

import numpy
import numpy as np


class SmartSync:
    def __init__(self, ws, number_of_nodes, logger: logging):
        self.logger = logger
        self.ws = ws
        self.number_of_nodes = number_of_nodes
        self.collector1 = np.full([ws, number_of_nodes + 1], np.NaN)  # First position represents the int_div of ws
        self.collector1[:, 0] = 0

    #  node_number should start at 0
    #  Returns all values as an numpy array for the current timestamp if it is full, else None
    def put(self, timestamp, node_number, value):
        node_number = node_number % self.number_of_nodes
        arr_pos = timestamp % self.ws
        int_div = timestamp // self.ws
        if self.collector1[arr_pos][0] < int_div:  # Clear row if cycled through array
            self.collector1[arr_pos][0] = int_div
            self.logger.debug("Smartsync clearing row " + str(arr_pos))
            for i in range(self.number_of_nodes):
                self.collector1[arr_pos][i + 1] = np.NaN
        elif self.collector1[arr_pos][0] > int_div:  # If int_div is from previous cycle
            return None

        self.collector1[arr_pos][node_number + 1] = value
        self.logger.debug("Smartsync value added to [" + str(arr_pos) + "][" + str(node_number + 1) + "]")
        self.logger.debug(
            str(np.count_nonzero(~np.isnan(self.collector1[arr_pos][1:]))) + " of " + str(self.number_of_nodes)
            + " values are filled in the row")

        if self.__collector_row_filled(self.collector1[arr_pos][1:]):
            self.logger.info("Row " + str(arr_pos) + " filled, returning values")
            return self.collector1[arr_pos][1:]
        return None

    @staticmethod
    def __collector_row_filled(row):
        return not numpy.isnan(row).any()
