from data_processor import DataProcessor
import pandas as pd
from datetime import datetime
from collections import deque
from timeit import default_timer as timer
import numpy as np
from time import sleep
import csv
import gc
import decimal
import sys
import os

def build_input_row(stock, data_processors, time, normalize):
    stack = []
    min_max_tuple = None
    #for stock in stocks:
    market_orders = np.array((data_processors[stock].get_window())).ravel()
    if(len(market_orders)>0):
        if normalize:
            market_orders, min_max_tuple = normalize_array(market_orders)
        stack.extend(market_orders)
        

    #for stock in stocks:
    financial_models = list(data_processors[stock].get_financial_models())
    if(len(financial_models)>0):
        stack.extend(financial_models)
    stack.append(int(time))
    return stack, min_max_tuple
    
    
def to_norm(x, min_x, max_x):
    max_x = max_x#* 1.10
    min_x = min_x#* 0.9
    return round((x-min_x)/(max_x-min_x+0.0001),4)
    
def from_norm(x, min_x, max_x):
    max_x = max_x#/ 1.10
    min_x = min_x#/ 0.9
    return round(x*(max_x-min_x+0.0001)+min_x,4)
    
def normalize_array(array):
    max_x = array.max() #* 1.10
    min_x = array.min() #* 0.9
    
    f = lambda x: np.around((x-min_x)/(max_x-min_x+0.0001),4)
  
    return f(array), (min_x, max_x)

def is_market_open(time):
    dt = datetime.fromtimestamp(time)
    is_open = dt.weekday() <= 4
    is_open = is_open and (dt.hour >= 9 and dt.hour < 18 and (False if dt.hour == 17 and dt.minute >= 30 else True))
    return is_open

def gen_rows(df):
    for row in df.itertuples(index=False):
        yield row._asdict()

def get_column_names(stock, params):
    cols = []
    for i in range(params["window_size"]):
        for feature in params["market_order_features"]:
            cols.append(stock+"-"+feature+"-"+str(i))
    for model in params["financial_models"]:
        cols.append(stock+model)
    cols.append("ts")
    return cols

def clear_data_processors(data_processors):
    for _,dp in data_processors.items():
        dp.clear()

def is_market_day_over(time):
    dt = datetime.fromtimestamp(time)
    return dt.hour >= 17 and dt.minute >= 30

def find_next_time(time_price_map, from_time, end_time):
    while True:
        from_time += 1
        if from_time in time_price_map:
            return from_time
        elif from_time >= end_time:
            return -1

def get_up_down_target(cur_price, fut_price, threshold):
    delta = threshold*cur_price
    if(fut_price <= cur_price+delta and fut_price >= cur_price-delta):
        return 1
    elif(fut_price < cur_price):
        return 0
    else:
        return 2

def generate_x_name(params):
    name = "x_" + params["stock"] + "_"
    if(params["window_size"] > 0):
        name += str(params["window_size"]) + "_"
        for mof in params["market_order_features"]:
            name += mof[0]
    else:
        fm = params["financial_models"][0]
        name += fm
    name += ".csv"
    return name

def generate_y_name(params):
    name = "y_" + params["stock"] + "_"
    if(params["window_size"] > 0):
        name += str(params["window_size"])
    name += ".csv"
    return name

def create_y_data(time_price_map, start_time, end_time, params, min_max_map):
    normalize = params["normalize"]

    print("Creating y data...")
    print("Map size: " + str(len(time_price_map)))
    current_time = int(start_time)
    time_jumps = [15, 30, 60, 300, 600]
    dir_path = params["stock"] + "/"
    name = generate_y_name(params)
    file = open(dir_path+name, 'w+', newline ='')

    with file:
        write = csv.writer(file, delimiter=';')
        rows_head = ["15s", "15ud", "30s", "30ud", "60s", "60ud", "300s", "300ud", "600s", "600ud", "ts"]
        if normalize:
            rows_head.insert(-1, "min")
            rows_head.insert(-1, "max", )
        write.writerow(rows_head) #TODO
        print("Looping price map ...")
        while current_time + time_jumps[-1] <= end_time:
            row = []
            if current_time in time_price_map:
                cur_price = time_price_map[current_time]
                
            for jump in time_jumps:
                t = current_time + jump
                if t in time_price_map:
                    fut_price = time_price_map[t]
                    if normalize:
                        mini, maxi = min_max_map[t]
                        fut_price = from_norm(fut_price, mini, maxi) #Unnormalize
                        #print("Unnormalized price y "+ str(fut_price))
                        mini, maxi = min_max_map[current_time]
                        fut_price = to_norm(fut_price, mini, maxi) #Normalize with the min and max from cur_time
                        #print("Normalized price y with x min,max "+ str(fut_price))
                    row.append(fut_price)
                    row.append(get_up_down_target(cur_price, fut_price, params["threshold"]))
                else:
                    current_time = find_next_time(time_price_map, t, end_time)
                    break
            if current_time == -1:
                return
            elif len(row) == 2*len(time_jumps):
                mini, maxi = min_max_map[current_time]
                row.append(mini)
                row.append(maxi)
                row.append(int(current_time))
                write.writerow(row)
                current_time += 1


def end_trade_day(write, data_processors, day):
    clear_data_processors(data_processors)
    pop_amount = min(len(day), 600)
    for i in range(pop_amount):
        day.pop()
    write.writerows(day)


def create_train_data(params, _data):
    start = timer()
    stock = params["stock"]
    normalize = params["normalize"]
    print("Calculating for: " + stock)
    filter = _data["stock"] == stock
    #for stock in params["stocks"]:
    #    filter = (data["stock"] == stock) | filter
    data = _data[filter]

    data_processors ={}
    print("Calc starting window ...")
    #for stock in params["stocks"]:
    dp = DataProcessor(stock, params)
    start_time = dp.process_start_window(data)
    data_processors[stock] = dp

    print(start_time)
    time = start_time
    data = data[data["publication_time"] >= time]

    market_orders = gen_rows(data)

    time_price_map = {}
    min_max_map = {}

    dir_path = stock + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    name = generate_x_name(params)
    file = open(dir_path+name, 'w+', newline ='')
    end_time = 0
    day = []
    with file:
        write = csv.writer(file, delimiter=';')
        write.writerow(get_column_names(stock, params))
        print("Processing market orders ...")
        for market_order in market_orders:
            if(market_order["publication_time"] > time):
                while(market_order["publication_time"] > time):
                    if not is_market_open(time):
                        time = market_order["publication_time"]
                        end_trade_day(write, data_processors, day)
                        day = []
                    elif data_processors[stock].is_window_filled():
                        row, min_max_tuple = build_input_row(stock, data_processors, time, normalize)
                        day.append(row)
                        end_time = row[-1]
                        if params["window_size"] != 0:   
                            time_price_map[time] = row[-2] #TODO NOT HARDCODE
                            if min_max_tuple!=None:
                                min_max_map[time] = min_max_tuple
                        time += 1
                    else:
                        time += 1
            data_processors[stock].process(market_order)
        print(time)
        end_trade_day(write, data_processors, day)
    if params["window_size"] != 0:
        create_y_data(time_price_map, start_time, end_time, params, min_max_map)

    end = timer()
    print("Time: "+str(end-start)+"s")


params = {
    "stocks" : ["Swedbank_A"],
    "window_sizes" : [500],
    "financial_models" : [],
    "market_order_features" : ["price"],
    "threshold" : 0.0002,
    "normalize" : True
}

datafile = sys.argv[1]
print("Reading csv: " + datafile)
data = pd.read_csv(datafile, sep=";", usecols=["price", "stock", "publication_time"])

for stock in params["stocks"]:
    param = {}
    param["financial_models"] = []
    param["threshold"] = 0.0002
    param["stock"] = stock
    param["normalize"] = params["normalize"]
    for ws in params["window_sizes"]:
        param["window_size"] = ws
        for mof in params["market_order_features"]:
            param["market_order_features"] = [mof]
            create_train_data(param, data)

    for fm in params["financial_models"]:
        param["financial_models"] = [fm]
        param["window_size"] = 0
        param["market_order_features"] = []
        create_train_data(param, data)

print("Done")
