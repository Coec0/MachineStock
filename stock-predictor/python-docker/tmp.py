import pandas as pd
#x_data = pd.read_csv("x.csv", sep=";")
print("Reading csv file...")
data = pd.read_csv("y_Swedbank_A_1.csv", sep=";", usecols=["600s", "ts"], squeeze=True)
#data = data.iloc[600:]


#x_data = x_data["Timestamp"]
print("Converting to int...")

#data = data.astype(int)
#x_data.to_csv("x_ts.csv", index=False, sep = ';')
data.to_csv("blaYY.csv", index=False, sep = ';')
print("Done")
