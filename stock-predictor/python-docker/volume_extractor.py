import pandas as pd

stock = "Swedbank_A"

print("Reading csv ...")
df = pd.read_csv("market_orders_apr.csv", sep=";", usecols=["stock", "publication_time", "volume"])
df = df[df["stock"] == stock]
del df["stock"]
print(df.head())
df.columns = ["ts", "volume"]


def gen_rows(df):
    for row in df.itertuples(index=False):
        yield row._asdict()


file = open(stock+"/"+stock+"_volume.csv", "w")
file.write("volume;ts\n")
file.close()

time = 0
vol_count = 0
rows = gen_rows(df)
file = open(stock+"/"+stock+"_volume.csv", "a")
for row in rows:
    if time == 0:
        time = row["ts"]
    while row["ts"] != time:
        file.write(str(vol_count) + ";" + str(time) + "\n")
        time += 1
        vol_count = 0
    vol_count += row["volume"]
file.close()
print("Done")

