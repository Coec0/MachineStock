from request_omx import RequestOMX
import re
import sys
import json
import os
import time
import itertools
import threading
import pytz
from pytz import timezone, all_timezones
from threading import Thread
from datetime import date, datetime, time as time2, timedelta

def wait_until(end_datetime):
    while True:
        end_datetime = end_datetime.astimezone(timezone('Europe/Stockholm'))
        diff = (end_datetime - datetime.now(tz=pytz.timezone('Europe/Stockholm'))).total_seconds()
        if diff < 0: return       # In case end_datetime was in past to begin with
        time.sleep(diff/2)
        if diff <= 0.1: return


def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.now(tz=pytz.timezone('Europe/Stockholm')).time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

def sleep_if_market_is_closed(id):
    if is_time_between(time2(17,31), time2(7,59)): #Stock market is closed, bug makes it of by one hour, therefore 7,59 is used instead of 8,59
        if datetime.now(tz=pytz.timezone('Europe/Stockholm')).weekday() == 4: #It's friday
            tomorrow = datetime.now(tz=pytz.timezone('Europe/Stockholm')) + timedelta(days=3)
        else:
            tomorrow = datetime.now(tz=pytz.timezone('Europe/Stockholm')) + timedelta(days=1)
        print("Stock closed, time to sleep, id:"+str(id))
        wait_until(datetime(tomorrow.year, tomorrow.month, tomorrow.day, 8, 0, 0, 0)) #Stock market is closed, bug makes it of by one hour, therefore 8 is used instead of 9
        print("Stock open, starting to work, id:"+str(id))

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

def stock_thread(selected_stock, selected_section, lock):
    prev = ""
    while True:
        try:
            date_ = date.today().strftime("%Y-%m-%d")
            dir_path = "data/limitorders/"+date_+"/"+selected_section+"/"

            lock.acquire()
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            lock.release()

            data = fetcher.fetch_limitorders(selected_stock["id"])
            if(data.status_code == 200):
                write_header = False
                file_path = dir_path+selected_stock["name"]+".csv"
                if not os.path.exists(file_path):
                    write_header = True
                file = open(file_path, "a", encoding="utf-8")
                if write_header:
                    file.write(csv_header + '\n')

                try:
                    if re.sub(r'^.*?;', ';', str(prev)) != re.sub(r'^.*?;', ';', str(parse_order(json.loads(data.text)))):
                        parsed = parse_order(json.loads(data.text))
                        file.write(parse_order(json.loads(data.text)) + '\n')
                        prev = parsed
                except:
                     pass #There were no limit orders or something was malformed, ignore

                sleep_if_market_is_closed(selected_stock["id"])
                time.sleep(1)
        except:
            pass #Some kind of error happened, probably a problem with the web request. Just continue

stocks_json_file = str(sys.argv[1])

print("Time started " + str(datetime.now(tz=pytz.timezone('Europe/Stockholm')).time())) # To make sure that the correct timezone
with open(stocks_json_file, "r", encoding="utf-8") as file:
    stocks_dict = json.loads(file.read())

csv_header = "sep=;\ntimestamp;bid1;bid2;bid3;bid4;bid5;ask1;ask2;ask3;ask4;ask5;bid_volume1;bid_volume2;bid_volume3;bid_volume4;bid_volume5;ask_volume1;ask_volume2;ask_volume3;ask_volume4;ask_volume5"

fetcher = RequestOMX()

index = 0
for key in stocks_dict: #Loop until correct position
    lock = threading.Lock()
    section = stocks_dict[key]
    for stock in section:
        Thread(target=stock_thread, args=(stock, key, lock, )).start()
