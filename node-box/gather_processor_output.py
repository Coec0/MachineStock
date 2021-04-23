from processors.ema_processor import EMAProcessor
from processors.rsi_processor import RSIProcessor
from processors.macd_processor import MACDProcessor
from processors.volatility_processor import VolatilityProcessor
import numpy as np
import pandas as pd
# Financial indicators - features: price, ema, rsi, macd, volatility, channels
data_file = "x_Swedbank_A_1_p_fullnormalized.csv"

df = pd.read_csv(data_file, sep=";")
df = df[df["ts"] >= 1614788464]
df = df[df["ts"] <= 1615800583]

print(df.head())

def gen_rows(df):
    for row in df.itertuples(index=False):
        yield row._asdict()


data_generator = gen_rows(df)

dir_path = "layer2-train/data/"

processors = [#("ema30", EMAProcessor(30, True)),
              ("rsi60", RSIProcessor(60, True))]
              #("macd", MACDProcessor()),
              #("volatility30", VolatilityProcessor(30))

for n, p in processors:
    file = open(dir_path+"/"+n+".csv", "w")
    file.write(n+";ts\n")
    file.close()

for d in data_generator:
    for n, p in processors:
        ts, output = p.process(int(d["ts"]), np.array([float(d["SwedbankPrice"])]))
        file = open(dir_path + "/" + n + ".csv", "a")
        file.write(str(output)+";"+str(ts)+"\n")
        file.close()
