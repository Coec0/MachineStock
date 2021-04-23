from processors.ema_processor import EMAProcessor
from processors.rsi_processor import RSIProcessor
from processors.macd_processor import MACDProcessor
from processors.volatility_processor import VolatilityProcessor
from processors.channels_processor import ChannelsProcessor
import numpy as np
import pandas as pd
# Financial indicators - features: price, ema, rsi, macd, volatility, channels
data_file = "x_Swedbank_A_1_p_fullnormalized.csv"

df = pd.read_csv(data_file, sep=";")
df = df[df["ts"] >= 1614585602]

print(df.head())

def gen_rows(df):
    for row in df.itertuples(index=False):
        yield row._asdict()


data_generator = gen_rows(df)

dir_path = "layer2-train/data/"

processors = [["channels20min", ["channel_k_min_1200", "channel_k_max_1200","channel_m_min_1200", "channel_m_max_1200"], ChannelsProcessor(1200, True)],
              ["channels2hour", ["channel_k_min_7200", "channel_k_max_7200","channel_m_min_7200", "channel_m_max_7200"], ChannelsProcessor(7200, True)]]
              #("ema30", EMAProcessor(30, True)),
              #("rsi60", RSIProcessor(60, True)),
              #("macd", MACDProcessor()),
              #("volatility30", VolatilityProcessor(30))]

for n, hds, _ in processors:
    file = open(dir_path+"/"+n+".csv", "w")
    for h in hds:
        file.write(h+";")
    file.write("ts\n")
    file.close()

for d in data_generator:
    for n, _, p in processors:
        ts, outputs = p.process(int(d["ts"]), np.array([float(d["SwedbankPrice"])]))
        file = open(dir_path + "/" + n + ".csv", "a")
        for x in outputs:
            file.write(str(x)+";")
        file.write(str(ts)+"\n")
        file.close()
