from torch import float32


class Observer:
    def notify(self, result: (int, float32), start_time=-1):
        """Notify the observer. start_time is used for benchmarking and should be -1 if not used. """
        pass
