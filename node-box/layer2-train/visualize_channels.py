import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from functools import reduce

channel_data = pd.read_csv("data/channels1day.csv", sep=";")
price_data = pd.read_csv("../x_Swedbank_A_1_p_fullnormalized.csv", sep=";", usecols=["SwedbankPrice", "ts"])

data = pd.merge(channel_data, price_data, how="inner", on=["ts"])

x_data_price = np.asarray(data["SwedbankPrice"])
x_data_min_k = np.asarray(data["channel_k_min_86400"])
x_data_max_k = np.asarray(data["channel_k_max_86400"])
price_channel_min = np.asarray(data["channel_m_min_86400"])
price_channel_max = np.asarray(data["channel_m_max_86400"])

print(x_data_min_k)
print(price_channel_min)

plt.plot(list(range(len(price_channel_min))), price_channel_min)
plt.plot(list(range(len(price_channel_min))), price_channel_max)
plt.plot(list(range(len(x_data_price))), x_data_price)
axes = plt.gca()
axes.set_ylim([0.6,1])
plt.show()
