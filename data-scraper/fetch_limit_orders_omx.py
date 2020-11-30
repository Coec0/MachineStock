from request_omx import RequestOMX
import re
import sys
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

if len(sys.argv) != 2: #This is actually 1 argument, as the filename counts as an argument
	sys.exit("Missing launch parameter for which stock to choose 0-385")

selected_stock = int(sys.argv[1])

with open("stocks.json", "r") as file:
    stocks_dict = json.loads(file.read())

stocks_length = 0
for key in stocks_dict:
    stocks_length += len(stocks_dict[key])
    
if selected_stock >= stocks_length or selected_stock < 0:
    sys.exit("Launch parameter for which stock to choose is wrong. Selected a value between 0-"+str(stocks_length-1))

csv_header = "sep=;\ntimestamp;bid1;bid2;bid3;bid4;bid5;ask1;ask2;ask3;ask4;ask5;bid_volume1;bid_volume2;bid_volume3;bid_volume4;bid_volume5;ask_volume1;ask_volume2;ask_volume3;ask_volume4;ask_volume5"

today = date.today()
now = time.localtime()

date = today.strftime("%Y-%m-%d")
current_time = time.strftime("%H:%M:%S", now)

fetcher = RequestOMX()

index = 0
for key in stocks_dict: #Loop until correct position
    section = stocks_dict[key]
    for stock in section:
        if index == selected_stock:
            selected_stock = stock
            selected_section = key
            break
        else:
            index +=1
    else:
        continue
    break


dir_path = "limitorders/"+date+"/"+selected_section+"/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

prev = ""
while True:
    data = fetcher.fetch_limitorders(selected_stock["id"])
    if(data.status_code == 200):
        write_header = False
        file_path = dir_path+stock["name"]+".csv"
        if not os.path.exists(file_path):
            write_header = True
        file = open(file_path, "a")
        if write_header:
            file.write(csv_header + '\n')
        if re.sub(r'^.*?;', ';', str(prev)) != re.sub(r'^.*?;', ';', str(parse_order(json.loads(data.text)))):
            try:
                parsed = parse_order(json.loads(data.text))
                file.write(parse_order(json.loads(data.text)) + '\n')
                prev = parsed
            except:
                 pass #There were no limit orders or something was malformed, ignore
        time.sleep(1)
