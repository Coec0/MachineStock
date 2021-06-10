from datetime import datetime

import pandas as pd

import auto_train_layer2 as atl
import io
import torch.nn.functional as f
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
# Måste loopa igenom alla kombinatortion här
# Skapa större fil med alla financial indicators
# Koda early stopping


def main():
    financial_indicators = ["ema30", "ema15","macd", "rsi5", "rsi30","volatility100", "volatility50"]
    predictors_swedbank = ["SwedbankPred70", "SwedbankPred200", "SwedbankPred700"]
    predictors_nordea = ["NordeaPred70", "NordeaPred200", "NordeaPred700"]
    learning_rates = [0.001] #, 0.0001, 0.000001

    for nordea in predictors_nordea:
        for lr in learning_rates:
            for fi in financial_indicators:
                x_cols = predictors_swedbank.copy()
                x_cols.append(fi)
                if nordea != "None":
                    x_cols.append(nordea)
                model_3 = atl.CombinerModel_3(len(x_cols))
                model_lr = atl.LinReg(len(x_cols))
                name_3 = model_3.__class__.__name__ + "_" + fi + "_lr" + str(lr) + "_" + nordea
                name_lr = model_lr.__class__.__name__ + "_" + fi + "_lr" + str(lr) + "_" + nordea
                print("Running: " + name_3)
                run_model(model_3, name_3, lr, nordea, "data/auto/resultfile.csv", x_cols)
                #print("Running: " + name_lr)
                #run_model(model_lr, name_lr, lr, nordea, "data/auto/resultfile.csv", x_cols)


            for i in range(len(financial_indicators)):
                tmp = financial_indicators.pop()
                fin_inds_str = str(financial_indicators)
                financial_indicators.insert(0, tmp)
                x_cols = predictors_swedbank.copy()
                x_cols.extend(financial_indicators)
                if nordea != "None":
                    x_cols.append(nordea)
                model_3 = atl.CombinerModel_3(len(x_cols))
                model_lr = atl.LinReg(len(x_cols))
                name_3 = model_3.__class__.__name__ + "_" + fin_inds_str + "_lr" + str(lr) + "_" + nordea
                name_lr = model_lr.__class__.__name__ + "_" + fin_inds_str + "_lr" + str(lr) + "_" + nordea
                print("Running: " + name_3)
                run_model(model_3, name_3, lr, nordea, "data/auto/resultfile.csv", x_cols)
                #print("Running: " + name_lr)
                #run_model(model_lr, name_lr, lr, nordea, "data/auto/resultfile.csv", x_cols)


def run_model(model, name, lr, nordea,  resultfile, x_cols):
    data = pd.read_csv("data/Swedbank_A_dist2_train_zscore_Nordea.csv", sep=";")
    train_data_loader, eval_data_loader, test_data_loader, test_data_y = split_data(data, x_cols)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    time = str(datetime.now().timestamp())
    atl.train_model(model, train_data_loader, eval_data_loader, loss_fn, optimizer, 500, "data/auto/logs/"+name+"_log_"+str(time)+".txt")
    torch.save(model.state_dict(), "data/auto/models/"+name+"._state_dict.pt")
    loss, preds_ = atl.evaluate_model(test_data_loader, model, loss_fn)

    preds, target = denorm(preds_, test_data_y)
    mse_loss_preds, mae_loss_preds, mse_loss_10min, mae_loss_10min, mse_loss_offset, mae_loss_offset = calc_losses(preds, target, test_data_y["avg"])

    with io.open(resultfile, "a", encoding="utf-8") as f:
        row = name+";"+nordea+";"+str(lr)+";"+"{:.5f}".format(mse_loss_preds)+";"+"{:.5f}".format(mae_loss_preds)+";"+"{:.5f}".format(mse_loss_10min)+";"+"{:.5f}".format(mae_loss_10min)+";"+"{:.5f}".format(mse_loss_offset)+";"+"{:.5f}".format(mae_loss_offset)
        f.write(row+"\n")


def from_norm(x, avg, stdev):
    return x * (stdev+0.000001) + avg


def denorm(preds_, test_data_y):
    preds = []
    target = []
    test_data_y_ = test_data_y.to_numpy()
    for i in range(len(test_data_y_)):
        (t, avg, stdev) = test_data_y_[i]
        target.append(from_norm(t, avg, stdev))
        preds.append(from_norm(preds_[i], avg, stdev))
    return preds, target


def calc_losses(preds, target, avg_y):
    target_tensor = torch.tensor(target, dtype=torch.float32)
    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()
    mse_loss_preds = loss_fn_mse(preds_tensor, target_tensor).item()
    mae_loss_preds = loss_fn_mae(preds_tensor, target_tensor).item()

    avg_10min = torch.tensor(avg_y.values, dtype=torch.float32)
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()
    mse_loss_10min = loss_fn_mse(avg_10min, target_tensor).item()
    mae_loss_10min = loss_fn_mae(avg_10min, target_tensor).item()

    target_for_offset = torch.tensor(target[:-30], dtype=torch.float32)
    offset = torch.tensor(target[30:], dtype=torch.float32)
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()
    mse_loss_offset = loss_fn_mse(offset, target_for_offset).item()
    mae_loss_offset = loss_fn_mae(offset, target_for_offset).item()

    return mse_loss_preds, mae_loss_preds, mse_loss_10min, mae_loss_10min, mse_loss_offset, mae_loss_offset


def split_data(_data, x_columns, batch_size=512):
    train_dev_split_index = round(0.8 * len(_data))
    dev_test_split_index = round(0.9 * len(_data))
    train_data = _data.loc[:train_dev_split_index]
    eval_data = _data.loc[train_dev_split_index:dev_test_split_index]
    test_data = _data.loc[dev_test_split_index:]

    train_data_x = torch.tensor(train_data[x_columns].values, dtype=torch.float32)
    train_data_y = torch.tensor(train_data[["target"]].values, dtype=torch.float32)

    train_data_t = TensorDataset(train_data_x, train_data_y)
    train_data_loader = DataLoader(train_data_t, batch_size=batch_size)

    eval_data_x = torch.tensor(eval_data[x_columns].values, dtype=torch.float32)
    eval_data_y = torch.tensor(eval_data[["target"]].values, dtype=torch.float32)
    eval_data_t = TensorDataset(eval_data_x, eval_data_y)
    eval_data_loader = DataLoader(eval_data_t, batch_size=batch_size)

    test_data_x = torch.tensor(test_data[x_columns].values, dtype=torch.float32)
    test_data_y = torch.tensor(test_data[["target"]].values, dtype=torch.float32)
    test_data_t = TensorDataset(test_data_x, test_data_y)
    test_data_loader = DataLoader(test_data_t, batch_size=batch_size)

    return train_data_loader, eval_data_loader, test_data_loader, test_data[["target", "avg", "stdev"]]


main()