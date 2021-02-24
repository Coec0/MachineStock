from data_processor import DataProcessor
import pandas as pd
from datetime import datetime
from collections import deque
from timeit import default_timer as timer
import numpy as np
from time import sleep
import csv

def build_input_row(stocks, data_processors, time):
    stack = []
    for stock in stocks:
        market_orders = np.array(data_processors[stock].get_window()).ravel()
        stack.append(market_orders)

    for stock in stocks:
        financial_models = np.array(data_processors[stock].get_financial_models())
        if(financial_models != None):
            stack.append(financial_models)

    #stack.append([time])

    return np.hstack(stack)

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
    return cols

def create_train_data(input, params):
    start = timer()
    print("Reading csv ...")
    data = pd.read_csv(input, sep=";")

    filter = data["stock"] == params["stocks"][0]
    for stock in params["stocks"]:
        filter = (data["stock"] == stock) | filter
    data = data[filter]

    data_processors ={}
    print("Calc starting window ...")
    for stock in params["stocks"]:
        dp = DataProcessor(stock, params["window_size"], useVol = True, useExpAvgPrice=False)
        start_time = dp.process_start_window(data)
        data_processors[stock] = dp

    print(start_time)

    time = start_time

    data = data[data["publication_time"] >= time]

    market_orders = gen_rows(data)

    print("Processing market orders ...")
    rows = []
    for market_order in market_orders:
        if(market_order["publication_time"] > time):
            row = build_input_row(params["stocks"], data_processors, time)
            while(market_order["publication_time"] > time):
                if is_market_open(time): rows.append(row)
                time += 1

        data_processors[market_order["stock"]].process(market_order)

    print(time)
    print("Rows amount: " + str(len(rows)))
    print(rows[0])
    print(rows[200000])
    print(rows[-1])

    print("Saving to csv ...")
    df = pd.DataFrame(rows, columns=get_column_names(params))
    df.to_csv("x.csv", index=False, sep = ';')

    end = timer()
    print("Time: "+str(end-start)+"s")
    print("Done")


params = {
    "stocks" : ["Swedbank_A"],
    "window_size" : 20,
    "financial_models" : [],
    "market_order_features" : ["price", "volume"]
}
create_train_data("market_orders_sorted.csv", params)

# input - filepath for input data CSV file
# assume that CVS is sorted for publication time
# returns nparray with all data


# Fill create start queue with all start




# input - filepath for input data CSV file
# output - file name to save train data in
# returns string with path to file
#def create_train_data_file(input, parameters):