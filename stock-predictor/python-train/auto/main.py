import trainbase
from torch import nn
from torch import optim
import itertools
import os

# replace % for number
file_x_const = "Swedbank_A/x_Swedbank_A_%_p_ema_rsi_macd_volatility_channels.csv"
file_y_const = "Swedbank_A/y_Swedbank_A_%.csv"

loss_fn = nn.MSELoss()

models = [trainbase.DeepModel(), trainbase.ShallowModel()]
window_sizes = [70, 200, 700]
fin_inds = ["price", "ema", "rsi", "macd", "volatility", "channels"]
cols_y = [("5s", 0), ("15s", 2), ("30s", 4), ("60s", 6)]
epochs = [5, 10, 50, 100]
batch_sizes = [512]
learning_rates = [0.1, 0.01, 0.001, 0.0001]

iterator = itertools.product(models, epochs, batch_sizes, window_sizes, fin_inds, learning_rates, cols_y)

for params in iterator:
    print(params)
    cols_x = []
    model, epoch, batch_size, ws, fin_ind, lr, col_y_tup = params
    if(fin_ind == "price"):
        input_size = ws
    elif(fin_ind == "ema"):
        input_size = ws+2
        cols_x = [input_size, input_size+1]
    elif(fin_ind == "rsi"):
        input_size = ws+1
        cols_x = [input_size+2]
    elif(fin_ind == "macd"):
        input_size = ws+1
        cols_x = [input_size+3]
    elif(fin_ind == "volatility"):
        input_size = ws+1
        cols_x = [input_size+4]
    elif(fin_ind == "channels"):
        input_size = ws+8
        cols_x = list(range(input_size+5, input_size+13))

    model.instantiate(input_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=0.001)

    cols_x.extend(list(range(ws)))

    col_y_name, col_y = col_y_tup

    files_x = [file_x_const.replace("%", str(input_size))]
    files_y = [file_y_const.replace("%", str(input_size))]

    foldername = model.getName()+"_"+col_y_name+"_"+str(epoch)+"_"+str(batch_size)+"_"+fin_ind+"_"+str(lr)
    filepath = "Swedbank_A/"+str(ws)+"/"+foldername+"/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        trainbase.train(files_x, files_y, model, input_size, ws, loss_fn, optimizer, filepath, epoch, batch_size, cols_x, col_y)
