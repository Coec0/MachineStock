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


def start_node_box(layer, input_size, processor, tag, file=None, tag_to_pos=None, verbosity=logging.WARNING):
    NodeBox("localhost", 5501, layer, input_size, processor, tag, tag_to_pos, file, verbosity=verbosity,
            benchmark=True)


file_all = "x_Swedbank_A_1_p_fullnormalized.csv"
weight_file1 = "dist-models/Swedbank_A/layer1/70_Deep_30s_35_512_price_1e-06_True/model_dict.pt"
weight_file2 = "dist-models/Swedbank_A/layer1/200_Deep_30s_5_512_price_1e-06_False/model_dict.pt"
weight_file3 = "dist-models/Swedbank_A/layer1/700_Deep_30s_35_512_price_1e-06_False/model_dict.pt"
weight_file_layer2 = "dist-models/Swedbank_A/layer2/layer2_model_dist.pt"

_tag = sys.argv[1]
boxes = 1
#Coordinator(5501, boxes, FullyConnectedStrategy())

time.sleep(3)
processor2 = DeepNetworkProcessor(None, 200, False)
t2 = threading.Thread(target=start_node_box, args=(0, 1, processor2, [_tag], file_all))
t2.start()
