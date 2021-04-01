import trainbase_norm
import trainbase
from torch import nn
from torch import optim
import os
import pandas as pd
import shutil
from datetime import datetime
import traceback


# replace % for number


def run(iterator):
    file_x_const = "Swedbank_A/x_Swedbank_A_%_p_ema_rsi_macd_volatility_channels_time.csv"
    file_y_const = "Swedbank_A/y_Swedbank_A_%.csv"

    file_x_const_norm = "Swedbank_A/x_Swedbank_A_%_p_fullnormalized_ema_rsi_macd_volatility_channels_time.csv"
    file_y_const_norm = "Swedbank_A/y_Swedbank_A_%_fullnormalized.csv"

    loss_fn = nn.MSELoss()
    result_frame = pd.DataFrame(
        columns=["stock", "avg-train-loss", "avg-val-loss", "avg-test-loss", "val-r2", "test-r2", "epochs",
                 "learning-rate", "x-col", "y-col", "window-size", "batch-size", "use-time"])
    try:
        for params in iterator:
            print(params)
            cols_x = []
            model, epoch, batch_size, ws, fin_ind, lr, col_y_tup, use_time, normal, name = params
            mult = 2 if use_time else 1
            if fin_ind == "price":
                input_size = ws * mult
            elif fin_ind == "ema":
                input_size = ws * mult + 2
                cols_x = [ws * mult, ws * mult + 1]
            elif fin_ind == "rsi":
                input_size = ws * mult + 1
                cols_x = [ws * mult + 2]
            elif fin_ind == "macd":
                input_size = ws * mult + 1
                cols_x = [ws * mult + 3]
            elif fin_ind == "volatility":
                input_size = ws * mult + 1
                cols_x = [ws * mult + 4]
            elif fin_ind == "channels":
                input_size = ws * mult + 8
                cols_x = list(range(ws * mult + 5, ws * mult + 13))

            model.instantiate(input_size)
            optimizer = optim.AdamW(model.parameters(), lr=lr, eps=0.001)

            cols_x.extend(list(range(ws * mult)))

            col_y_name, col_y = col_y_tup

            selected_file_x_const = file_x_const_norm if normal else file_x_const
            selected_file_y_const = file_y_const_norm if normal else file_y_const

            files_x = [selected_file_x_const.replace("%", str(ws))]
            files_y = [selected_file_y_const.replace("%", str(ws))]

            foldername = model.getName() + "_" + col_y_name + "_" + str(epoch) + "_" + str(
                batch_size) + "_" + fin_ind + "_" + str(lr) + "_" + str(use_time)
            normalized = "_normalized" if normal else ""
            filepath = "Swedbank_A/" + str(ws) + "/"
            min_norm = 121.0
            max_norm = 165.9
            if not os.path.exists(filepath):
                os.makedirs(filepath)
                if normal:
                    result = trainbase_norm.train(files_x, files_y, model, input_size, ws, loss_fn, optimizer, filepath,
                                                  epoch,
                                                  batch_size, cols_x, col_y, min_norm, max_norm)
                else:
                    result = trainbase.train(files_x, files_y, model, input_size, ws, loss_fn, optimizer, filepath,
                                             epoch, batch_size, cols_x, col_y)
                train_loss, val_loss, test_loss, val_r2, test_r2 = result
                row = ["Swedbank_A", train_loss, val_loss, test_loss, val_r2, test_r2, epoch, lr, fin_ind, col_y_name,
                       ws, batch_size, int(use_time)]
                result_frame.loc[len(result_frame)] = row

        ts = str(int(datetime.now().timestamp()))
        result_frame.to_csv("Swedbank_A/Swedbank_A_" + name + ts + ".csv", index=False)
        print("Done")
    except:
        print("Error")
        traceback.print_exc()
        ts = str(int(datetime.now().timestamp()))
        result_frame.to_csv("Swedbank_A/Swedbank_A_" + name + ts + ".csv", index=False)
        shutil.rmtree(filepath)
