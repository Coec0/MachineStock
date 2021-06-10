import logging
import time
from threading import Thread
from stat_track import StatTrack
from numpy import ndarray
from smartsync.smart_sync import SmartSync
from processors.node_box_processor import NodeBoxProcessor
from observer import Observer


class InputHandler:
    def __init__(self, ws, input_size, tag_to_pos: dict, processor: NodeBoxProcessor,
                 observer: Observer, logger: logging):
        self.logger = logger
        self.number_of_features = input_size
        self.tag_to_pos = tag_to_pos
        self.smart_sync = SmartSync(ws, input_size, logger)
        self.benchmark_sync = SmartSync(ws, input_size, None)
        self.processor = processor
        self.observer = observer
        self.benchmark_stat_tracks = StatTrack()
        self.benchmark_waiting_track = [0]*input_size
        self.benchmark_waiting_stat_track = StatTrack()
        self.number_of_benchmarks_to_skip = 10

    def put(self, timestamp, values: list, tags: list, start_time=-1):
        arr = None
        benchmark_arr = None
        for i in range(len(values)):
            pos = self.tag_to_pos[tags[i]]
            arr = self.smart_sync.put(timestamp, pos, values[i])
            if start_time != -1:
                benchmark_arr = self.benchmark_sync.put(timestamp, pos, start_time)
                self.benchmark_waiting_track[pos % self.number_of_features] = time.time()
        if arr is not None:
            thread = Thread(target=self.__process_arr, args=(timestamp, arr, -1, benchmark_arr))
            thread.start()

    def put_all(self, timestamp, values: ndarray, start_time=-1):
        thread = Thread(target=self.__process_arr, args=(timestamp, values, start_time))
        thread.start()

    def __process_arr(self, timestamp, arr, start_time=-1, start_times=None):
        self.observer.notify(self.processor.process(int(timestamp), arr), start_time=start_time)
        if start_times is not None:
            if self.number_of_benchmarks_to_skip == 0:
                current_time = time.time()
                for count, time_start in enumerate(start_times):
                    self.benchmark_stat_tracks.add(current_time - time_start)
                    self.benchmark_waiting_stat_track.add(current_time - self.benchmark_waiting_track[count])
                if len(self.benchmark_stat_tracks) % (5*self.number_of_features) == 0:
                    longest, shortest, mean, deviation = self.benchmark_stat_tracks.get()
                    print("Longest time: "+str(longest))
                    print("Shortest time: "+str(shortest))
                    print("Mean time: "+str(mean))
                    print("Standard deviation: " + str(deviation) + "\n")
                    longest, shortest, mean, deviation = self.benchmark_waiting_stat_track.get()
                    print("Waiting times in smart-sync:")
                    print("Longest time: " + str(longest))
                    print("Shortest time: " + str(shortest))
                    print("Mean time: " + str(mean))
                    print("Standard deviation: " + str(deviation) + "\n")
            else:
                self.number_of_benchmarks_to_skip -= 1

