from data_processor import DataProcessor
import pandas as pd
from datetime import datetime
from collections import deque
from timeit import default_timer as timer
import numpy as np
from time import sleep
import csv
import gc

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


    stack = [ '%.4f' % elem for elem in stack ]
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

    print("Processing market orders ...")
    rows = []
    for market_order in market_orders:
        stock = market_order["stock"]

        if(market_order["publication_time"] > time):
            while(market_order["publication_time"] > time):
                if not is_market_open(time):
                    clear_data_processors(data_processors)
                    time = market_order["publication_time"]
                elif data_processors[stock].is_window_filled():
                    #rows.append([int(time)])
                    row = build_input_row(params["stocks"], data_processors, time)
                    rows.append(row)
                    time += 1
                else:
                    time += 1
        data_processors[stock].process(market_order)
    print(time)
    print("Rows amount: " + str(len(rows)))


    #print("Converting to pandas dataframe ...")
    #del data
    #gc.collect() #Free memory
    #df = pd.DataFrame(rows, columns=get_column_names(params))
    s = params["stocks"][0]
    w = str(params["window_size"])

    fms = ""
    for f in params["financial_models"]:
        fms = fms + "_" + f

    name = "x_" + stock + "_" + w + "_p" + fms
    now = datetime.now().strftime("%H_%M")
    #print("Saving to csv ...")
    #df.to_csv("x_"+now+".csv", index=False, sep = ';')

    #data = [['Geeks'], [4], ['geeks !']]

    # opening the csv file in 'w+' mode
    file = open(name+".csv", 'w+', newline ='')

    rows.insert(0, get_column_names(params))

    # writing the data into the file
    with file:
        write = csv.writer(file, delimiter=';')
        write.writerows(rows)

    end = timer()
    print("Time: "+str(end-start)+"s")
    print("Done")

params1 = {
    "stocks" : ["Swedbank_A"],
    "window_size" : 10,
    "financial_models" : ["macd"],
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
