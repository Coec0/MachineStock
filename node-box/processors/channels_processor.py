from processors.node_box_processor import NodeBoxProcessor
from numpy import ndarray
from processors.price_channels import PriceChannels


class ChannelsProcessor(NodeBoxProcessor):

    def __init__(self, time_window, normalize):
        self.channel = PriceChannels(time_window, 10, normalize)

    def process(self, timestamp, features: ndarray) -> (int, list):
        self.channel.update({"price": features[0], "publication_time": timestamp})
        min_k, max_k = self.channel.get_min_max_k()
        min_m, max_m = self.channel.get_price_channel_min_max()
        return timestamp, [min_k, max_k, min_m, max_m]
