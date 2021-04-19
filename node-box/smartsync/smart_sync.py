import numpy as np


class SmartSync:
    def __init__(self, ws, number_of_nodes):
        self.ws = ws
        self.number_of_nodes = number_of_nodes
        self.collector1 = np.zeros([ws, number_of_nodes + 1])  # First position represents the int_div of ws

    #  node_number should start at 0
    #  Returns all values as an numpy array for the current timestamp if it is full, else None
    def put(self, timestamp, node_number, value):
        node_number = node_number % self.number_of_nodes
        arr_pos = timestamp % self.ws
        int_div = timestamp // self.ws
        if self.collector1[arr_pos][0] < int_div:  # Clear row if cycled through array
            self.collector1[arr_pos][0] = int_div
            for i in range(self.number_of_nodes):
                self.collector1[arr_pos][i + 1] = 0
        elif self.collector1[arr_pos][0] > int_div:  # If int_div is from previous cycle
            return None

        self.collector1[arr_pos][node_number + 1] = value

        if self.__collector_row_filled(self.collector1[arr_pos][1:]):
            return self.collector1[arr_pos][1:]
        return None

    @staticmethod
    def __collector_row_filled(row):
        for item in row:
            if item == 0:
                return False
        return True
