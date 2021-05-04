from collections import deque
from statistics import mean, pstdev


class StatTrack:
    def __init__(self):
        self.max_val = float('-inf')
        self.min_val = float('inf')
        self.numbers = deque()

    def __len__(self):
        return len(self.numbers)

    def add(self, number):
        if number > self.max_val:
            self.max_val = number
        if number < self.min_val:
            self.min_val = number
        self.numbers.append(number)

    def get(self):
        return self.max_val, self.min_val, mean(self.numbers), pstdev(self.numbers)

