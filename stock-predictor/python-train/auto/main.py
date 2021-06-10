import trainbase_norm
import trainbase
from torch import nn
from torch import optim
import os
import pandas as pd
import shutil
from datetime import datetime
import traceback


def run(stock, iterator):
    file_x_ = {}
    file_y_ = {}
    file_x_["Swedbank_A"] = "Swedbank_A/x_Swedbank_A_%_zscore$.csv"
    file_y_["Swedbank_A"] = "Swedbank_A/y_Swedbank_A_%_zscore.csv"
    file_x_["Nordea_Bank_Abp"] = "Nordea_Bank_Abp/x_Nordea_Bank_Abp_%_zscore$.csv"
    file_y_["Nordea_Bank_Abp"] = "Nordea_Bank_Abp/y_Nordea_Bank_Abp_%_zscore.csv"


    result_frame = pd.DataFrame(
        columns=["stock", "avg-train-loss", "avg-val-loss", "avg-test-loss", "mse_loss_preds", "mae_loss_preds", "mse_loss_10min", "mae_loss_10min", "mse_loss_offset", "mae_loss_offset", "epochs",
                 "learning-rate", "x-col", "y-col", "window-size", "batch-size", "use-time"])
    try:
        for params in iterator:
            print(params)
            cols_x = []
            model, epoch, batch_size, (ws, use_time), fin_ind, lr, col_y_tup, name = params
            mult = 2 if use_time else 1
            if fin_ind == "price":
                input_size = ws * mult

            model.instantiate(input_size)
            optimizer = optim.AdamW(model.parameters(), lr=lr, eps=0.001)

            cols_x.extend(list(range(ws * mult)))
            col_y_name, col_y = col_y_tup
            file_x = file_x_[stock].replace("%", str(ws)).replace("$", "_time" if use_time else "")
            file_y = file_y_[stock].replace("%", str(ws)).replace("$", "_time" if use_time else "")

            foldername = model.getName() + "_" + col_y_name + "_" + str(epoch) + "_" + str(
                batch_size) + "_" + fin_ind + "_" + str(lr) + "_" + str(use_time)

            filepath = stock + "/" + str(ws) + "/" + foldername + "/"
            loss_fn = nn.MSELoss()
            if not os.path.exists(filepath):
                os.makedirs(filepath)
                result = trainbase_norm.train(file_x, file_y, model, input_size, ws, loss_fn, optimizer, filepath,
                                              epoch,
                                              batch_size, cols_x)
                train_loss, val_loss, test_loss, mse_loss_preds, mae_loss_preds, mse_loss_10min, mae_loss_10min, mse_loss_offset, mae_loss_offset = result
                row = [stock, train_loss, val_loss, test_loss, mse_loss_preds, mae_loss_preds, mse_loss_10min, mae_loss_10min, mse_loss_offset, mae_loss_offset, epoch, lr, fin_ind, col_y_name,
                       ws, batch_size, int(use_time)]
                result_frame.loc[len(result_frame)] = row

        ts = str(int(datetime.now().timestamp()))
        result_frame.to_csv(stock+"/"+stock+"_" + name + ts + ".csv", index=False)
        print("Done")
    except:
        print("Error")
        traceback.print_exc()
        ts = str(int(datetime.now().timestamp()))
        result_frame.to_csv(stock+"/"+stock+"_" + name + ts + ".csv", index=False)
        #shutil.rmtree(filepath)
