from zscore_tracker import ZScoreTracker
import pandas as pd
from datetime import datetime
from timeit import default_timer as timer
import numpy as np
import csv
import os


def build_input_row(params, zscore_tracker, time):

    market_orders, market_times, avg, stdev = zscore_tracker.get_window()
    stack = market_orders

    if len(market_orders) > 0:
        if params["use-time"]:
            stack.extend(market_times)

    stack.append(avg)
    stack.append(stdev)
    stack.append(int(time))
    return stack, avg, stdev


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
            cols.append(stock+"-price-"+str(i))
    if params["use-time"]:
        for i in range(params["window_size"]):
            cols.append(stock+"-time-"+str(i))
    cols.append("avg")
    cols.append("stdev")
    cols.append("ts")
    return cols


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


def generate_x_name(params):
    name = "x_" + params["stock"] + "_"
    name += str(params["window_size"])
    name += "_zscore"
    if params["use-time"]:
        name += "_time"
    name += ".csv"
    return name


def generate_y_name(params):
    name = "y_" + params["stock"] + "_"
    name += str(params["window_size"])
    name += "_zscore"
    name += ".csv"
    return name


def to_norm(val, avg, stdev):
    return (val-avg) / stdev


def create_y_data(time_price_map, start_time, end_time, params, norm_map):
    print("Creating y data...")
    print("Map size: " + str(len(time_price_map)))
    current_time = int(start_time)
    time_jumps = [30]
    dir_path = params["stock"] + "/"
    name = generate_y_name(params)
    file = open(dir_path+name, 'w+', newline='')
    write = csv.writer(file, delimiter=';')
    rows_head = ["30s", "avg", "stdev", "ts"]
    write.writerow(rows_head)
    print("Looping price map ...")

    while current_time + 30 <= end_time:
        row = []
        for jump in time_jumps:
            t = current_time + jump
            if t in time_price_map:
                fut_price = time_price_map[t]
                row.append("{:.6f}".format(fut_price))
            else:
                current_time = find_next_time(time_price_map, t, end_time)
                break
        if current_time == -1:
            return
        elif len(row) == len(time_jumps):
            avg, stdev = norm_map[current_time]
            row.append(avg)
            row.append(stdev)
            row.append(int(current_time))
            write.writerow(row)
            current_time += 1
    file.close()


def end_trade_day(write, zscore_tracker, day, window_size):
    zscore_tracker.clear()
    pop_amount = min(len(day), 30)
    for i in range(pop_amount):
        day.pop()
    for r in range(len(day)):
        for i in range(window_size):
            day[r][i] = "{:.6f}".format(day[r][i])
    write.writerows(day)


def create_train_data(params, data):
    start = timer()
    stock = params["stock"]
    print("Calc starting window ...")

    zscore_tracker = ZScoreTracker(10, params)

    start_time = zscore_tracker.process_start_window(data)
    time = start_time
    data = data[data["publication_time"] >= time]
    market_orders = gen_rows(data)

    time_price_map = {}
    norm_map = {}

    dir_path = stock + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    name = generate_x_name(params)
    file = open(dir_path+name, 'w+', newline='')
    end_time = 0
    day = []
    write = csv.writer(file, delimiter=';')
    write.writerow(get_column_names(stock, params))
    print("Processing market orders ...")
    day_counter = 0
    latest_price = 0
    for market_order in market_orders:
        row_built = False
        if market_order["publication_time"] > time:
            while market_order["publication_time"] > time:
                if not is_market_open(time):
                    time = market_order["publication_time"]
                    end_trade_day(write, zscore_tracker, day, params["window_size"])
                    day = []
                    day_counter = 0
                elif zscore_tracker.is_window_filled():
                    if row_built:
                        copy_row = []
                        for d in row_:
                            copy_row.append(d)
                        copy_row[-1] = time
                        row = copy_row
                    else:
                        row_, avg, st_dev = build_input_row(params, zscore_tracker, time)
                        row_built = True
                        row = row_
                    day_counter += 1
                    if day_counter > 5:
                        time_price_map[time] = row[params["window_size"] - 1]
                        norm_map[time] = (avg, st_dev)
                        day.append(row)
                    zscore_tracker.add_data(latest_price)
                    end_time = row[-1]
                    time += 1
                else:
                    time += 1

        zscore_tracker.process(market_order)
        latest_price = market_order["price"]
    print(time)
    end_trade_day(write, zscore_tracker, day, params["window_size"])
    file.close()
    print("Endtime: " + str(end_time))
    create_y_data(time_price_map, start_time, end_time, params, norm_map)

    end = timer()
    print("Time: "+str(end-start)+"s")


params = {
    "stocks": ["Swedbank_A", "Nordea_Bank_Abp"],
    "window_sizes": [1],
    "use-time": False
}

datafile = "market_orders_apr.csv"
print("Reading csv: " + datafile)
data = pd.read_csv(datafile, sep=";", usecols=["price", "stock", "publication_time"])

for stock in params["stocks"]:
    param = {"stock": stock}
    param["use-time"] = params["use-time"]
    for ws in params["window_sizes"]:
        param["window_size"] = ws
        filter_ = data["stock"] == stock
        data_ = data[filter_]
        del data_["stock"]
        create_train_data(param, data_)

print("Done")
