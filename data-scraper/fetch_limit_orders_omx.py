from request_omx import RequestOMX
import json
import os
import time
import itertools
from datetime import date

def parse_order(raw_data):
	timestamp = raw_data['@ts']
	orders = raw_data['inst']['pd']['pl']
	bids = []
	asks = []
	bids_volume = []
	asks_volume = []

	for order in orders:
	    bids.append(order['@b'])
	    asks.append(order['@a'])
	    bids_volume.append(order['@bv'])
	    asks_volume.append(order['@av'])

	combined = list(itertools.chain(bids,asks,bids_volume,asks_volume))
	return(timestamp +';'+(';'.join(combined)))

csv_header = "sep=;\ntimestamp;bid1;bid2;bid3;bid4;bid5;ask1;ask2;ask3;ask4;ask5;bid_volume1;bid_volume2;bid_volume3;bid_volume4;bid_volume5;ask_volume1;ask_volume2;ask_volume3;ask_volume4;ask_volume5"

today = date.today()
now = time.localtime()

date = today.strftime("%Y-%m-%d")
current_time = time.strftime("%H:%M:%S", now)

fetcher = RequestOMX()

with open("stocks.json", "r") as file:
    stocks_dict = json.loads(file.read())

for key in stocks_dict:
    dir_path = "limitorders/"+date+"/"+key+"/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    section = stocks_dict[key]
    for stock in section:
        data = fetcher.fetch_limitorders(stock["id"])
        if(data.status_code == 200):
            write_header = False
            file_path = dir_path+stock["name"]+".csv"
            if not os.path.exists(file_path):
                write_header = True
            file = open(file_path, "a")
            if write_header:
                file.write(csv_header + '\n')
            try:
                file.write(parse_order(json.loads(data.text)) + '\n')
            except:
                 pass #There were no limit orders or something was malformed, ignore
            #time.sleep(0.5)
