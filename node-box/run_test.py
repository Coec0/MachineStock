from coordinator import Coordinator
from coordinator_strategies.fully_connected_strategy import FullyConnectedStrategy
from nodebox import NodeBox
import threading
from processors.deep_network_processor import DeepNetworkProcessor
from processors.combiner_processor import CombinerProcessor


def start_node_box(layer, input_size, processor, file=None):
    if file is None:
        NodeBox("localhost", 5501, layer, 3, processor, file)
    else:
        NodeBox("localhost", 5501, layer, input_size, processor, file)


file_all = "x_Swedbank_A_1_p_fullnormalized.csv"
weight_file1 = "dist-models/Swedbank_A/layer1/70_Deep_30s_35_512_price_1e-06_True/model_dict.pt"
weight_file2 = "dist-models/Swedbank_A/layer1/200_Deep_30s_5_512_price_1e-06_False/model_dict.pt"
weight_file3 = "dist-models/Swedbank_A/layer1/700_Deep_30s_35_512_price_1e-06_False/model_dict.pt"
Coordinator(5501, 4, FullyConnectedStrategy())
processor1 = DeepNetworkProcessor(weight_file1, 30, 140, True)
t1 = threading.Thread(target=start_node_box, args=(0, 140, processor1, file_all))
t1.start()
processor2 = DeepNetworkProcessor(weight_file2, 30, 200, False)
t2 = threading.Thread(target=start_node_box, args=(0, 200, processor2, file_all))
t2.start()
processor3 = DeepNetworkProcessor(weight_file3, 30, 700, False)
t3 = threading.Thread(target=start_node_box, args=(0, 700, processor3, file_all))
t3.start()

processor_final = CombinerProcessor(None, 0, 3)
start_node_box(1, 3, processor_final)
