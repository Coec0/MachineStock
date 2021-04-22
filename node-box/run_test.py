from coordinator import Coordinator
from coordinator_strategies.fully_connected_strategy import FullyConnectedStrategy
from nodebox import NodeBox
import threading
from deep_network_processor import DeepNetworkProcessor


def start_node_box(layer, weight_file, input_size, timestamp_future, file=None):
    processor = DeepNetworkProcessor(weight_file, timestamp_future, input_size)
    if file is None:
        NodeBox("localhost", 5500, layer, 3, processor, file)
    else:
        NodeBox("localhost", 5500, layer, input_size, processor, file)


file1 = "x_Swedbank_A_70_p_fullnormalized_ema_rsi_macd_volatility_channels_time.csv"
file2 = "x_Swedbank_A_200_p_fullnormalized_ema_rsi_macd_volatility_channels_time.csv"
file3 = "x_Swedbank_A_700_p_fullnormalized_ema_rsi_macd_volatility_channels_time.csv"
weight_file1 = "dist-models/Swedbank_A/layer1/70_Deep_30s_35_512_price_1e-06_True/model_dict.pt"
weight_file2 = "dist-models/Swedbank_A/layer1/200_Deep_30s_5_512_price_1e-06_False/model_dict.pt"
weight_file3 = "dist-models/Swedbank_A/layer1/700_Deep_30s_35_512_price_1e-06_False/model_dict.pt"
Coordinator(5500, 4, FullyConnectedStrategy())
t1 = threading.Thread(target=start_node_box, args=(0, weight_file1, 140, 30, file1))
t1.start()
t2 = threading.Thread(target=start_node_box, args=(0, weight_file2, 200, 30, file2))
t2.start()
t3 = threading.Thread(target=start_node_box, args=(0, weight_file3, 700, 30, file3))
t3.start()
start_node_box(1, None, 3, 0)
#l1_0.connect("localhost", 12348)
#l1_1 = NodeBox(12346, "1", 1)

#l2 = NodeBox(12347, "2", 2)