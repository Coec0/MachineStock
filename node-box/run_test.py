import logging

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
    NodeBox("localhost", 5501, layer, input_size, processor, tag, tag_to_pos, file, verbosity=verbosity)


file_all = "x_Swedbank_A_1_p_fullnormalized.csv"
weight_file1 = "dist-models/Swedbank_A/layer1/70_Deep_30s_35_512_price_1e-06_True/model_dict.pt"
weight_file2 = "dist-models/Swedbank_A/layer1/200_Deep_30s_5_512_price_1e-06_False/model_dict.pt"
weight_file3 = "dist-models/Swedbank_A/layer1/700_Deep_30s_35_512_price_1e-06_False/model_dict.pt"
weight_file_layer2 = "dist-models/Swedbank_A/layer2/layer2_model_dist.pt"

Coordinator(5501, 10, FullyConnectedStrategy())
processor1 = DeepNetworkProcessor(weight_file1, 140, True)
t1 = threading.Thread(target=start_node_box, args=(0, 2, processor1, ["price1"], file_all))
t1.start()
processor2 = DeepNetworkProcessor(weight_file2, 200, False)
t2 = threading.Thread(target=start_node_box, args=(0, 1, processor2, ["price2"], file_all))
t2.start()
processor3 = DeepNetworkProcessor(weight_file3, 700, False)
t3 = threading.Thread(target=start_node_box, args=(0, 1, processor3, ["price3"], file_all))
t3.start()

# Financial indicators - features: price, ema, rsi, macd, volatility, channels
processor4 = EMAProcessor(30, True)
t4 = threading.Thread(target=start_node_box, args=(0, 1, processor4, ["ema"], file_all))
t4.start()
processor5 = RSIProcessor(30, True)
t5 = threading.Thread(target=start_node_box, args=(0, 1, processor5, ["rsi"], file_all))
t5.start()
processor6 = MACDProcessor()
t6 = threading.Thread(target=start_node_box, args=(0, 1, processor6, ["macd"], file_all))
t6.start()
processor7 = VolatilityProcessor(30)
t7 = threading.Thread(target=start_node_box, args=(0, 1, processor7, ["volatility"], file_all))
t7.start()
processor8 = ChannelsProcessor(1200, True)
t8 = threading.Thread(target=start_node_box, args=(0, 1, processor8,
                                                   ["channel_k_min_1200", "channel_k_max_1200",
                                                    "channel_m_min_1200", "channel_m_max_1200"], file_all))
t8.start()
processor9 = ChannelsProcessor(7200, True)
t9 = threading.Thread(target=start_node_box, args=(0, 1, processor9,
                                                   ["channel_k_min_7200", "channel_k_max_7200",
                                                    "channel_m_min_7200", "channel_m_max_7200"], file_all))
t9.start()

processor_final = CombinerProcessor(weight_file_layer2, 15)
start_node_box(1, 15, processor_final, ["final"], tag_to_pos={
    "price1": 0,
    "price2": 1,
    "price3": 2,
    "ema": 3,
    "rsi": 4,
    "macd": 5,
    "volatility": 6,
    "channel_k_min_1200": 7,
    "channel_k_max_1200": 8,
    "channel_m_min_1200": 9,
    "channel_m_max_1200": 10,
    "channel_k_min_7200": 11,
    "channel_k_max_7200": 12,
    "channel_m_min_7200": 13,
    "channel_m_max_7200": 14,
}, verbosity=logging.INFO)
