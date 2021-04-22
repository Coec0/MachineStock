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


def start_node_box(layer, input_size, processor, file=None):
    NodeBox("localhost", 5501, layer, input_size, processor, file)


file_all = "x_Swedbank_A_1_p_fullnormalized.csv"
weight_file1 = "dist-models/Swedbank_A/layer1/70_Deep_30s_35_512_price_1e-06_True/model_dict.pt"
weight_file2 = "dist-models/Swedbank_A/layer1/200_Deep_30s_5_512_price_1e-06_False/model_dict.pt"
weight_file3 = "dist-models/Swedbank_A/layer1/700_Deep_30s_35_512_price_1e-06_False/model_dict.pt"
Coordinator(5501, 8, FullyConnectedStrategy())
processor1 = DeepNetworkProcessor(weight_file1, 140, True)
t1 = threading.Thread(target=start_node_box, args=(0, 2, processor1, file_all))
t1.start()
processor2 = DeepNetworkProcessor(weight_file2, 200, False)
t2 = threading.Thread(target=start_node_box, args=(0, 1, processor2, file_all))
t2.start()
processor3 = DeepNetworkProcessor(weight_file3, 700, False)
t3 = threading.Thread(target=start_node_box, args=(0, 1, processor3, file_all))
t3.start()

# Financial indicators - features: price, ema, rsi, macd, volatility, channels
processor4 = EMAProcessor(30, True)
t4 = threading.Thread(target=start_node_box, args=(0, 1, processor4, file_all))
t4.start()
processor5 = RSIProcessor(30, True)
t5 = threading.Thread(target=start_node_box, args=(0, 1, processor5, file_all))
t5.start()
processor6 = MACDProcessor()
t6 = threading.Thread(target=start_node_box, args=(0, 1, processor6, file_all))
t6.start()
processor7 = VolatilityProcessor(30)
t7 = threading.Thread(target=start_node_box, args=(0, 1, processor7, file_all))
t7.start()

processor_final = CombinerProcessor(None, 7)
start_node_box(1, 7, processor_final)
