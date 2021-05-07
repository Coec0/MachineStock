import logging
import sys
import time

from coordinator import Coordinator
from coordinator_strategies.fully_connected_strategy import FullyConnectedStrategy
from nodebox import NodeBox
import threading
from processors.deep_network_processor import DeepNetworkProcessor
from processors.combiner_processor import CombinerProcessor
from processors.ema_processor import EMAProcessor
from processors.rsi_processor import RSIProcessor
from processors.macd_processor import MACDProcessor
from processors.volatility_processor import VolatilityProcessor
from processors.channels_processor import ChannelsProcessor

""" This test runs many node-boxes in parallel in layer1,
 and one node-box in layer 2 receives everything"""


def start_node_box(layer, input_size, processor_local, tag_local, file=None, tag_to_pos=None, verbosity=logging.CRITICAL):
    NodeBox("localhost", 5501, layer, input_size, processor_local, tag_local, tag_to_pos, file, verbosity=verbosity,
            benchmark=True)


if len(sys.argv) > 1:
    layer_1_size = int(sys.argv[1])
else:
    layer_1_size = 4
Coordinator(5501, layer_1_size + 1, FullyConnectedStrategy(), logging.DEBUG)
file_all = "x_Swedbank_A_1_p_fullnormalized.csv"
weight_file = "dist-models/Swedbank_A/layer1/70_Deep_30s_35_512_price_1e-06_True/model_dict.pt"
processor = DeepNetworkProcessor(weight_file, 140, True)
tag_to_pos_layer2 = {}
for i in range(layer_1_size):
    tag = "price"+str(i+1)
    tag_to_pos_layer2[tag] = i

weight_file_layer2 = "dist-models/Swedbank_A/layer2/layer2_model_dist.pt"
processor_final = CombinerProcessor(None, layer_1_size)
start_node_box(1, layer_1_size, processor_final, ["final"], tag_to_pos=tag_to_pos_layer2, verbosity=logging.WARNING)
