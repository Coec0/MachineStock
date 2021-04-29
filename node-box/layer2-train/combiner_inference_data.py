import pandas as pd

df1 = pd.read_csv("../x_Swedbank_A_1_p_fullnormalized.csv", sep=";", usecols=["SwedbankPrice", "ts"])
df2 = pd.read_csv("../Swedbank_A_volume.csv", sep=";")

#df70 = pd.read_csv("data/70time_normal_preds.csv", sep=";")
#df200 = pd.read_csv("data/200_normal_preds.csv", sep=";")
#df700 = pd.read_csv("data/700_normal_preds.csv", sep=";")

#ema30 = pd.read_csv("data/ema30.csv", sep=";")
#macd = pd.read_csv("data/macd.csv", sep=";")
#rsi60 = pd.read_csv("data/rsi60.csv", sep=";")
#rsi5 = pd.read_csv("data/rsi5.csv", sep=";")
#vol100 = pd.read_csv("data/volatility100.csv", sep=";")
#channel2h = pd.read_csv("data/channels2hour.csv", sep=";")
#channel20 = pd.read_csv("data/channels20min.csv", sep=";")

# For swedbank min_norm = 121.0
# For swedbank max_norm = 165.9

result = pd.merge(df1, df2, how="inner", on=["ts"])
#result = pd.merge(result, ema30, how="inner", on=["ts"])
#result = pd.merge(result, macd, how="inner", on=["ts"])
#result = pd.merge(result, rsi5, how="inner", on=["ts"])
#result = pd.merge(result, vol100, how="inner", on=["ts"])
#result = pd.merge(result, channel2h, how="inner", on=["ts"])
#result = pd.merge(result, channel20, how="inner", on=["ts"])
#result = result[["pred70", "pred200", "pred700", "ema30", "macd", "rsi5", "volatility100",
#                 "channel_k_min_1200", "channel_k_max_1200", "channel_m_min_1200", "channel_m_max_1200",
#                 "channel_k_min_7200", "channel_k_max_7200", "channel_m_min_7200", "channel_m_max_7200", "target", "ts"]]

result = result[["SwedbankPrice", "volume", "ts"]]
result.to_csv("../Swedbank_A_vol_price.csv", sep=";", index=False)

print("Done")