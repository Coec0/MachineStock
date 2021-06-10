from processors.ema_processor import EMAProcessor
from processors.rsi_processor import RSIProcessor
from processors.macd_processor import MACDProcessor
from processors.obv_processor import OBVProcessor
from processors.volatility_processor import VolatilityProcessor
from processors.channels_processor import ChannelsProcessor
from processors.klinger_processor import KlingerProcessor
import numpy as np
import pandas as pd
# Financial indicators - features: price, ema, rsi, macd, volatility, channels
data_file = "x_Swedbank_A_1_zscore_time.csv"

df = pd.read_csv(data_file, sep=";", usecols=["SwedbankPrice", "ts", "SwedbankAvg", "SwedbankStdev"])
df = df[df["ts"] >= 1614585602]

def gen_rows(df):
    for row in df.itertuples(index=False):
        yield row._asdict()

def from_norm(x, avg, stdev):
    return x * (stdev+0.000001) + avg


# ["obv1", ["obv1"], OBVProcessor(1)]
# ["klinger", ["klinger"], KlingerProcessor()]
dir_path = "layer2-train/data/"

processors = [#["channels30min_tmp", ["channel_k_min_1800", "channel_k_max_1800","channel_m_min_1800", "channel_m_max_1800"], ChannelsProcessor(1800, True)],
              #["channels2hour_tmp", ["channel_k_min_7200", "channel_k_max_7200","channel_m_min_7200", "channel_m_max_7200"], ChannelsProcessor(7200, True)]
              ["rsi30", ["rsi30"], RSIProcessor(30)],
              ["ema15", ["ema15"], EMAProcessor(15, True)],
              #["macd", ["macd"], MACDProcessor()],
              ["volatility50", ["volatility50"], VolatilityProcessor(50)]
             ]

for n, hds, _ in processors:
    file = open(dir_path+n+".csv", "w")
    for h in hds:
        file.write(h+";")
    file.write("ts\n")
    file.close()

for n, _, p in processors:
    print("Working on " + n)
    output = ""
    data_generator = gen_rows(df)
    for d in data_generator:
        price = float(from_norm(d["SwedbankPrice"], d["SwedbankAvg"], d["SwedbankStdev"]))
        time = int(d["ts"])
        ts, outputs = p.process(time, [price])
        for x in outputs:
            output += str(x) + ";"
        output += str(ts) + "\n"
    file = open(dir_path + n + ".csv", "a")
    file.write(output)
    file.close()
