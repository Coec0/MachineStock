import pandas as pd
import sys
from threading import Thread
import subprocess
from target_calculator import TargetCalculator

def run(window_size):
    target_calculator = TargetCalculator(stock, data, window_size)
    list_items = []
    while True:
      next = target_calculator.get()
      list_items.append(next)
      if len(list_items) % 10000 == 0:
        print(len(list_items))
      if not target_calculator.step():
        break

    print(len(list_items))

    df = pd.DataFrame(list_items, columns=column_items)
    df.to_csv("y_" + stock + "_" + str(window_size) + ".csv", index=False, sep = ';')

stock = "Swedbank_A"
data = pd.read_csv("market_orders_sorted.csv", sep=";")#, nrows=1000000)
column_items = ["15s", "15sa", "15ud", "30s", "30sa", "30ud", "45s", "45sa", "45ud", "60s", "60sa", "60ud", "180s", "180sa", "180ud", "300s", "300sa", "300ud", "600s", "600sa", "600ud", "ts"]

threads = []

for i in range(len(sys.argv) - 1):
    t = Thread(target=run, args=(int(sys.argv[i+1]),))
    threads.append(t)
    t.start()
    
for x in threads:
     x.join()




  

