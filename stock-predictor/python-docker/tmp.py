import pandas as pd
#x_data = pd.read_csv("x.csv", sep=";")
data = pd.read_csv("x_Nordea_SEB_Swedbank_70_p.csv", sep=";")

#x_data = x_data["Timestamp"]
data = data["ts"]
data["ts"] = data["ts"].astype(int)
#x_data.to_csv("x_ts.csv", index=False, sep = ';')
data.to_csv("x_ts_Nordea_SEB_Swedbank_70_p.csv", index=False, sep = ';')
print("Done")
