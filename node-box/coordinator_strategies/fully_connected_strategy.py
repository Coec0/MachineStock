from coordinator_strategies.strategy import Strategy


class FullyConnectedStrategy(Strategy):

    def execute(self, layer_dict: dict):
        """ Executes the chosen strategy"""
        keys_size = len(layer_dict.keys())

        for i in range(len(layer_dict[0])):
            layer_dict[0][i]["server_ip_port"] = []

        for i in range(1, keys_size):
            for j in range(len(layer_dict[i])):
                layer_dict[i][j]["server_ip_port"] = []
                for node_above in layer_dict[i - 1]:
                    layer_dict[i][j]["server_ip_port"].append((node_above["local_ip"], node_above["port"]))
