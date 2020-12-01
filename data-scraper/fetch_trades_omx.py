from request_omx import RequestOMX
import json
import os
import time

from_date = "2020-11-18"
fetcher = RequestOMX()

with open("stocks.json", "r") as file:
    stocks_dict = json.loads(file.read())

for key in stocks_dict:
    dir_path = "data/marketorders/"+from_date+"/"+key+"/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    section = stocks_dict[key]
    for stock in section:
        data = fetcher.fetch_marketorders(stock["id"], from_date)
        if(data.status_code == 200):
            file = open(dir_path+"/"+stock["name"]+".csv", "w")
            file.write(data.text)
            time.sleep(0.5)
