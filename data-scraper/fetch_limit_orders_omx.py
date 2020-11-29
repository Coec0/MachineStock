from request_omx import RequestOMX
import json
import os
import time
from datetime import date

today = date.today()
now = time.localtime()

date = today.strftime("%Y-%m-%d")
current_time = time.strftime("%H:%M:%S", now)

fetcher = RequestOMX()

with open("stocks.json", "r") as file:
    stocks_dict = json.loads(file.read())

for key in stocks_dict:
    dir_path = "limitorders/"+date+"/"+current_time+"/"+key+"/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    section = stocks_dict[key]
    for stock in section:
        data = fetcher.fetch_limitorders(stock["id"])
        if(data.status_code == 200):
            file = open(dir_path+"/"+stock["name"]+".json", "w")
            file.write(data.text)
            #time.sleep(0.5)
