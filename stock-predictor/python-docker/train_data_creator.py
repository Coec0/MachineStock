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
        stack.append(market_orders)

    for stock in stocks:
        financial_models = np.array(data_processors[stock].get_financial_models())
        if(len(financial_models)>0):
            stack.append(financial_models)

    stack.append([time])

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
<<<<<<< HEAD
        dp = DataProcessor(stock, params)
=======
        dp = DataProcessor(stock, params["window_size"], useVol = False, useExpAvgPrice=False)
>>>>>>> 743f6446195f1dee768be844f5f2fe26ec59fc47
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
<<<<<<< HEAD
=======

            
>>>>>>> 743f6446195f1dee768be844f5f2fe26ec59fc47
            while(market_order["publication_time"] > time):
                if not is_market_open(time):
                    clear_data_processors(data_processors)
                    time = market_order["publication_time"]
                elif data_processors[stock].is_window_filled():
<<<<<<< HEAD
                    #rows.append([int(time)])
                    row = build_input_row(params["stocks"], data_processors, time)
=======
                    row = build_input_row(params["stocks"], data_processors, time)
                    time += 1
>>>>>>> 743f6446195f1dee768be844f5f2fe26ec59fc47
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
    now = datetime.now().strftime("%H_%M")
    #print("Saving to csv ...")
    #df.to_csv("x_"+now+".csv", index=False, sep = ';')

    #data = [['Geeks'], [4], ['geeks !']] 

    # opening the csv file in 'w+' mode 
    file = open("x_"+now+".csv", 'w+', newline ='')
    
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
<<<<<<< HEAD
    "window_size" : 1,
    "financial_models" : ["rsi"],
=======
    "window_size" : 500,
    "financial_models" : [],
>>>>>>> 743f6446195f1dee768be844f5f2fe26ec59fc47
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
