import pandas as pd

df70 = pd.read_csv("data/70time_normal_preds.csv", sep=";")
df200 = pd.read_csv("data/200_normal_preds.csv", sep=";")
df700 = pd.read_csv("data/700_normal_preds.csv", sep=";")

ema30 = pd.read_csv("data/ema30.csv", sep=";")
macd = pd.read_csv("data/macd.csv", sep=";")
rsi60 = pd.read_csv("data/rsi60.csv", sep=";")
vol30 = pd.read_csv("data/volatility30.csv", sep=";")
channel2h = pd.read_csv("data/channels2hour.csv", sep=";")
channel20 = pd.read_csv("data/channels20min.csv", sep=";")

# For swedbank min_norm = 121.0
# For swedbank max_norm = 165.9

result = pd.merge(df70, df200, how="inner", on=["ts", "target"])
result = pd.merge(result, df700, how="inner", on=["ts", "target"])
result = pd.merge(result, ema30, how="inner", on=["ts"])
result = pd.merge(result, macd, how="inner", on=["ts"])
result = pd.merge(result, rsi60, how="inner", on=["ts"])
result = pd.merge(result, vol30, how="inner", on=["ts"])
result = pd.merge(result, channel2h, how="inner", on=["ts"])
result = pd.merge(result, channel20, how="inner", on=["ts"])
result = result[["pred70", "pred200", "pred700", "ema30", "macd", "rsi60", "volatility30",
                 "channel_k_min_1200", "channel_k_max_1200", "channel_m_min_1200", "channel_m_max_1200",
                 "channel_k_min_7200", "channel_k_max_7200", "channel_m_min_7200", "channel_m_max_7200", "target", "ts"]]

result.to_csv("data/dist-L2-traindata.csv", sep=";", index=False)

print("Done")