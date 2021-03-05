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

def build_input_row(stocks, data_processors, time):
    stack = []
    for stock in stocks:
        market_orders = np.array(data_processors[stock].get_window()).ravel()
        if(len(market_orders)>0):
            stack.extend(market_orders)

    for stock in stocks:
        financial_models = np.array(data_processors[stock].get_financial_models())
        if(len(financial_models)>0):
            stack.extend(financial_models)

    for i in range(len(stack)):
        d = decimal.Decimal(str(stack[i]))
        if d.as_tuple().exponent < -4:
            stack[i] = '%.4f' % stack[i]

    stack.append(int(time))
    return stack

def is_market_open(time):
    dt = datetime.fromtimestamp(time)
    is_open = dt.weekday() <= 4
    is_open = is_open and (dt.hour >= 9 and dt.hour < 18 and (False if dt.hour == 17 and dt.minute >= 30 else True))
    return is_open

def gen_rows(df):
    for row in df.itertuples(index=False):
        yield row._asdict()

def get_column_names(params):
    cols = []
    for i in range(params["window_size"]):
        for stock in params["stocks"]:
            for feature in params["market_order_features"]:
                cols.append(stock+"-"+feature+"-"+str(i))
    for stock in params["stocks"]:
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


def create_y_data(time_price_map, start_time, end_time, stock, w):
    print("Creating y data...")
    current_time = start_time
    time_jumps = [15, 30, 60, 300, 600]
    name = "y_" + stock + "_" + w
    file = open(name+".csv", 'w+', newline ='')

    with file:

        write = csv.writer(file, delimiter=';')
        write.writerow(["15s", "30s", "60s", "300s", "600s"]) #TODO
        print("Looping price map ...")
        while current_time + time_jumps[-1] <= end_time:
            row = []
            for jump in time_jumps:
                t = current_time + jump
                if t in time_price_map:
                    price = time_price_map[t]
                    row.append(price)
                else:
                    current_time = find_next_time(time_price_map, t, end_time)
                    break
            if current_time == -1:
                return
            elif len(row) == len(time_jumps):
                write.writerow(row)






def create_train_data(input, params, data):
    start = timer()

    filter = data["stock"] == params["stocks"][0]
    for stock in params["stocks"]:
        filter = (data["stock"] == stock) | filter
    data = data[filter]

    data_processors ={}
    print("Calc starting window ...")
    for stock in params["stocks"]:
        dp = DataProcessor(stock, params)
        start_time = dp.process_start_window(data)
        data_processors[stock] = dp

    print(start_time)
    time = start_time
    data = data[data["publication_time"] >= time]

    market_orders = gen_rows(data)

    s = params["stocks"][0]
    w = str(params["window_size"])

    fms = ""
    for f in params["financial_models"]:
        fms = fms + "_" + f

    time_price_map = {}

    name = "x_" + stock + "_" + w + "_p" + fms
    file = open(name+".csv", 'w+', newline ='')
    end_time = 0
    i_tmp = 0
    with file:
        write = csv.writer(file, delimiter=';')
        write.writerow(get_column_names(params))
        print("Processing market orders ...")
        for market_order in market_orders:

            stock = market_order["stock"]

            if(market_order["publication_time"] > time):
                while(market_order["publication_time"] > time):

                    if not is_market_open(time):
                        clear_data_processors(data_processors)
                        time = market_order["publication_time"]
                    elif data_processors[stock].is_window_filled():
                        i_tmp += 1
                        row = build_input_row(params["stocks"], data_processors, time)
                        end_time = row[-1]
                        time_price_map[time] = row[-2]
                        write.writerow(row)
                        time += 1
                    else:
                        time += 1
            if i_tmp > 1000:
                break
            data_processors[stock].process(market_order)
        print(time)

    create_y_data(time_price_map, start_time, end_time, stock, w)

    end = timer()
    print("Time: "+str(end-start)+"s")
    print("Done")

params1 = {
    "stocks" : ["Swedbank_A"],
    "window_size" : 5,
    "financial_models" : [],
    "market_order_features" : ["price"]
}

print("Reading csv ...")
data = pd.read_csv("market_orders_sorted.csv", sep=";")
create_train_data("market_orders_sorted.csv", params1, data)
#create_train_data("market_orders_sorted.csv", params2, data)
#create_train_data("market_orders_sorted.csv", params3, data)
#create_train_data("market_orders_sorted.csv", params4, data)

# input - filepath for input data CSV file
# assume that CVS is sorted for publication time
# returns nparray with all data


# Fill create start queue with all start




# input - filepath for input data CSV file
# output - file name to save train data in
# returns string with path to file
#def create_train_data_file(input, parameters):
