import pandas as pd

#df1 = pd.read_csv("../x_Swedbank_A_1_p_fullnormalized.csv", sep=";", usecols=["SwedbankPrice", "ts"])
#df2 = pd.read_csv("../Swedbank_A_volume.csv", sep=";")

target = pd.read_csv("../y_Swedbank_A_1_zscore.csv", sep=";")
target["target"] = target["30s"]
del target["30s"]

df70 = pd.read_csv("data/Swedbank/70time_zscore_preds.csv", sep=";", usecols=["SwedbankPred70", "ts"])
df200 = pd.read_csv("data/Swedbank/200_zscore_preds.csv", sep=";", usecols=["SwedbankPred200", "ts"])
df700 = pd.read_csv("data/Swedbank/700_zscore_preds.csv", sep=";", usecols=["SwedbankPred700", "ts"])

df70_nordea = pd.read_csv("data/Nordea/70_zscore_preds.csv", sep=";", usecols=["NordeaPred70", "ts"])
df200_nordea = pd.read_csv("data/Nordea/200_zscore_preds.csv", sep=";", usecols=["NordeaPred200", "ts"])
df700_nordea = pd.read_csv("data/Nordea/700_zscore_preds.csv", sep=";", usecols=["NordeaPred700", "ts"])

ema30 = pd.read_csv("data/ema30.csv", sep=";")
ema15 = pd.read_csv("data/ema15.csv", sep=";")
macd = pd.read_csv("data/macd.csv", sep=";")
rsi5 = pd.read_csv("data/rsi5.csv", sep=";")
rsi30 = pd.read_csv("data/rsi30.csv", sep=";")
vol100 = pd.read_csv("data/volatility100.csv", sep=";")
vol50 = pd.read_csv("data/volatility50.csv", sep=";")
#channel2h = pd.read_csv("data/channels2hour.csv", sep=";")
#channel30 = pd.read_csv("data/channels30min.csv", sep=";")

# For swedbank min_norm = 121.0
# For swedbank max_norm = 165.9

result = pd.merge(target, df70, how="inner", on=["ts"])
result = pd.merge(result, df70_nordea, how="inner", on=["ts"])
result = pd.merge(result, df200_nordea, how="inner", on=["ts"])
result = pd.merge(result, df700_nordea, how="inner", on=["ts"])
result = pd.merge(result, df200, how="inner", on=["ts"])
result = pd.merge(result, df700, how="inner", on=["ts"])
result = pd.merge(result, ema30, how="inner", on=["ts"])
result = pd.merge(result, ema15, how="inner", on=["ts"])
result = pd.merge(result, macd, how="inner", on=["ts"])
result = pd.merge(result, rsi5, how="inner", on=["ts"])
result = pd.merge(result, rsi30, how="inner", on=["ts"])
result = pd.merge(result, vol100, how="inner", on=["ts"])
result = pd.merge(result, vol50, how="inner", on=["ts"])
#result = pd.merge(result, channel2h, how="inner", on=["ts"])
#result = pd.merge(result, channel30, how="inner", on=["ts"])
result = result[["SwedbankPred70", "SwedbankPred200", "SwedbankPred700", "NordeaPred70", "NordeaPred200", "NordeaPred700","ema30", "ema15","macd", "rsi5", "rsi30","volatility100", "volatility50", "target", "avg", "stdev", "ts"]]
                 #"channel_k_min_1800", "channel_k_max_1800", "channel_m_min_1800", "channel_m_max_1800",
                 #"channel_k_min_7200", "channel_k_max_7200", "channel_m_min_7200", "channel_m_max_7200", "target", "ts"]]


result.to_csv("data/Swedbank_A_dist2_train_zscore_Nordea.csv", sep=";", index=False)

print("Done")