import trainbase
from torch import nn
from torch import optim
import itertools
import os
import pandas as pd
import shutil
from datetime import datetime
import traceback

# replace % for number
file_x_const = "Swedbank_A/x_Swedbank_A_%_p_ema_rsi_macd_volatility_channels.csv"
file_y_const = "Swedbank_A/y_Swedbank_A_%.csv"

loss_fn = nn.MSELoss()

models = [trainbase.DeepModel(), trainbase.ShallowModel()]
window_sizes = [70, 200, 700]
fin_inds = ["price", "ema", "rsi", "macd", "volatility", "channels"]
cols_y = [("5s", 0), ("15s", 2), ("30s", 4), ("60s", 6)]
epochs = [5, 50]
batch_sizes = [512]
learning_rates = [0.01, 0.001, 0.0001]
useTime = [False]

iterator = itertools.product(models, epochs, batch_sizes, window_sizes, fin_inds, learning_rates, cols_y, useTime)

resultframe = pd.DataFrame(columns = ["stock", "avg-train-loss", "avg-val-loss", "avg-test-loss", "val-r2", "test-r2", "epochs", "learning-rate", "x-col", "y-col", "window-size", "batch-size", "use-time"])

try:
    for params in iterator:
        print(params)
        cols_x = []
        model, epoch, batch_size, ws, fin_ind, lr, col_y_tup, useTime = params
        mult = 2 if useTime else 1
        if(fin_ind == "price"):
            input_size = ws*mult
        elif(fin_ind == "ema"):
            input_size = ws*mult+2
            cols_x = [ws*mult, ws*mult+1]
        elif(fin_ind == "rsi"):
            input_size = ws*mult+1
            cols_x = [ws*mult+2]
        elif(fin_ind == "macd"):
            input_size = ws*mult+1
            cols_x = [ws*mult+3]
        elif(fin_ind == "volatility"):
            input_size = ws*mult+1
            cols_x = [ws*mult+4]
        elif(fin_ind == "channels"):
            input_size = ws*mult+8
            cols_x = list(range(ws*mult+5, ws*mult+13))

        model.instantiate(input_size)
        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=0.001)

        cols_x.extend(list(range(ws*mult)))

        col_y_name, col_y = col_y_tup

        files_x = [file_x_const.replace("%", str(ws))]
        files_y = [file_y_const.replace("%", str(ws))]

        foldername = model.getName()+"_"+col_y_name+"_"+str(epoch)+"_"+str(batch_size)+"_"+fin_ind+"_"+str(lr)+"_"+str(useTime)
        filepath = "Swedbank_A/"+str(ws)+"/"+foldername+"/"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            result = trainbase.train(files_x, files_y, model, input_size, ws, loss_fn, optimizer, filepath, epoch, batch_size, cols_x, col_y)
            train_loss, val_loss, test_loss, val_r2, test_r2 = result
            row = ["Swedbank_A", train_loss, val_loss, test_loss, val_r2, test_r2, epoch, lr, fin_ind, col_y_name, ws, batch_size, int(useTime)]
            resultframe.loc[len(resultframe)] = row

    ts = str(int(datetime.now().timestamp()))
    resultframe.to_csv("Swedbank_A/Swedbank_A_"+ts+".csv", index=False)
    print("Done")
except:
    print("Error")
    traceback.print_exc()
    ts = str(int(datetime.now().timestamp()))
    resultframe.to_csv("Swedbank_A/Swedbank_A_"+ts+".csv", index=False)
    shutil.rmtree(filepath)
