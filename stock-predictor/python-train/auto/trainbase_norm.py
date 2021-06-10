#!/usr/bin/env python
# coding: utf-8

# In[44]:
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import gc
import math
from sklearn.metrics import r2_score
import io
import os


#Params
dtype = torch.cuda.FloatTensor
device = torch.device('cuda:0')
chunksize = 1000000
data_split_ratio=0.8


def splitData(xs, ys, trainRatio):
    t = round(len(xs)*trainRatio)
    train_data_x = torch.tensor(xs[:t].values, dtype=torch.float32)
    train_data_y = torch.tensor(ys[:t].values, dtype=torch.float32)
    dev_data_x = torch.tensor(xs[t:].values, dtype=torch.float32)
    dev_data_y = torch.tensor(ys[t:].values, dtype=torch.float32)
    return TensorDataset(train_data_x, train_data_y), TensorDataset(dev_data_x, dev_data_y)

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()

    def instantiate(self, input_size):
        self.fc1 = nn.Linear(input_size, input_size*2).type(dtype)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2 = nn.Linear(input_size*2, round(input_size*1.5)).type(dtype)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3 = nn.Linear(round(input_size*1.5), round(input_size*0.5)).type(dtype)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        self.fc4 = nn.Linear(round(input_size*0.5), 20).type(dtype)
        self.fc4.weight.data.uniform_(-0.1, 0.1)
        self.fc5 = nn.Linear(20, 1).type(dtype)
        self.fc5.weight.data.uniform_(-0.1, 0.1)

    def getName(self):
        return "Deep"

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x= F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        return x

class ShallowModel(nn.Module):
    def __init__(self):
        super().__init__()

    def instantiate(self, input_size):
        self.fc1 = nn.Linear(input_size, input_size*3).type(dtype)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2 = nn.Linear(input_size*3, round(input_size*0.5)).type(dtype)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3 = nn.Linear(round(input_size*0.5, 1)).type(dtype)
        self.fc3.weight.data.uniform_(-0.1, 0.1)

    def getName(self):
        return "Shallow"

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

def evaluate_model(data, model, loss_fn):
    losses = []
    ys = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for x, y in data:
            y = y.type(dtype).squeeze()
            x = x.type(dtype)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            ys.extend(y.tolist() if len(y.size())>0 else [y])
            predictions.extend(pred.tolist() if len(pred.size())>0 else [pred])
        avg_loss = sum(losses)/len(losses)
    r2 = r2_score(ys, predictions)
    return avg_loss, predictions, r2

def train_model(model, train_data_loader, dev_data_loader, loss_fn, optimizer, epochrange, batchsize, filepath):
    log_file = io.open(filepath+"log.txt", "a", encoding="utf-8")
    train_avg_loss = 0
    dev_avg_loss = 0
    r2 = 0
    for epoch in range(epochrange):
        losses = []
        model.train()
        for x, y in train_data_loader:
            y = y.type(dtype).squeeze()
            x = x.type(dtype)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compute accuracy and loss in the entire training set
        train_avg_loss = sum(losses)/len(losses)
        dev_avg_loss,_,r2 = evaluate_model(dev_data_loader, model, loss_fn)

        # Display metrics
        display_str = 'Epoch {} '
        display_str += '\tLoss: {:.6f} '
        display_str += '\tLoss (val): {:.6f}'
        display_str += '\tR^2 score: {:.4f}'
        log_file.write(display_str.format(epoch, train_avg_loss, dev_avg_loss, r2)+"\n")
    log_file.close()
    return train_avg_loss, dev_avg_loss, r2


def train_chunk(model, loss_fn, optimizer, epochrange, x_data, y_data, data_split_ratio, batch_size, filepath):
    train_data, dev_data = splitData(x_data, y_data, data_split_ratio)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    dev_data_loader = DataLoader(dev_data, batch_size=batch_size, drop_last=True)
    return train_model(model, train_data_loader, dev_data_loader, loss_fn, optimizer, epochrange, batch_size, filepath)


def from_norm(x, avg, stdev):
    return x * (stdev+0.000001) + avg


def train(file_x, file_y, model, input_size, window_size, loss_fn, optimizer, filepath, epochrange, batch_size, cols_x):
    model = model.to(device)
    test_data_x = pd.DataFrame()
    test_data_y = pd.DataFrame()
    last_epoch_train_loss = []
    last_epoch_val_loss = []
    last_epoch_r2 = []
    print("Current file: " + file_x)
    total_rows = sum(1 for row in open(file_x, 'r'))
    number_of_loops = int(total_rows/chunksize)
    print("Number of chunks: " + str(number_of_loops))
    current_loop = 0
    with pd.read_csv(file_x, sep=";", dtype="float32", usecols = cols_x, chunksize=chunksize) as reader_x, pd.read_csv(file_y, sep=";", dtype="float32", chunksize=chunksize) as reader_y:
        for chunk_x, chunk_y in zip(reader_x, reader_y):
            print("Progress: " + "{:.2f}".format(100 * current_loop/number_of_loops) + "%")
            if current_loop < data_split_ratio * number_of_loops:
                train_avg_loss, dev_avg_loss, r2 = train_chunk(model, loss_fn, optimizer, epochrange, chunk_x, chunk_y["30s"], data_split_ratio, batch_size,filepath)
                last_epoch_train_loss.append(train_avg_loss)
                last_epoch_val_loss.append(dev_avg_loss)
                last_epoch_r2.append(r2)
            else:
                print("Append test data")
                test_data_x = test_data_x.append(chunk_x)
                test_data_y = test_data_y.append(chunk_y)
            current_loop += 1
            del chunk_x
            del chunk_y

    torch.save(model, filepath+"model.pt")
    test_data_x = torch.tensor(test_data_x.values, dtype=torch.float32)
    test_data_y_ = torch.tensor(test_data_y["30s"].values, dtype=torch.float32)
    test_data = TensorDataset(test_data_x, test_data_y_)

    x_avg = test_data_x[:, 0:window_size]
    x_avg = torch.mean(x_avg, 1)

    test_data_loader = DataLoader(test_data, batch_size=512)
    loss, preds_, r2 = evaluate_model(test_data_loader, model, loss_fn)



    x_avg = x_avg.tolist()
    target = []
    preds = []
    test_data_y_ = test_data_y.to_numpy()
    for i in range(len(test_data_y_)):
        (t, avg, stdev, _) = test_data_y_[i]
        p = preds_[i]
        target.append(from_norm(t, avg, stdev))
        preds.append(from_norm(p, avg, stdev))
        x_avg[i] = from_norm(x_avg[i], avg, stdev)

    target_tensor = torch.tensor(target, dtype=torch.float32)

    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()
    mse_loss_preds = loss_fn_mse(preds_tensor, target_tensor)
    mae_loss_preds = loss_fn_mae(preds_tensor, target_tensor)

    avg_10min = torch.tensor(test_data_y["avg"].values, dtype=torch.float32)
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()
    mse_loss_10min = loss_fn_mse(avg_10min, target_tensor)
    mae_loss_10min = loss_fn_mae(avg_10min, target_tensor)

    target_for_offset = torch.tensor(target[:-30], dtype=torch.float32)
    offset = torch.tensor(target[30:], dtype=torch.float32)
    loss_fn_mse = nn.MSELoss()
    loss_fn_mae = nn.L1Loss()
    mse_loss_offset = loss_fn_mse(offset, target_for_offset)
    mae_loss_offset = loss_fn_mae(offset, target_for_offset)


    with io.open(filepath+"log.txt", "a", encoding="utf-8") as f:
        f.write("\n------\nNorm Test loss: " + str(loss))


    ##plt.plot(list(range(159000, 160000)), preds[159000:160000], label="Predictions")
    #plt.plot(list(range(159000, 160000)), target[159000:160000], label="Target")
    #axes = plt.gca()
    #plt.legend()
    #plt.show()
    #plt.savefig(filepath+'zoom.pdf')
    #plt.close()

    plt.plot(list(range(len(preds))), preds, label="Predictions")
    plt.plot(list(range(len(preds))), target, label="Target")
    plt.plot(list(range(len(preds))), x_avg, label="Avg price")
    axes = plt.gca()
    plt.legend()
    #axes.set_xlim([100000,120000])
    plt.show()
    plt.savefig(filepath+'avg.pdf')
    plt.close()

    avg_train_loss = last_epoch_train_loss[-1]
    avg_val_loss = last_epoch_val_loss[-1]

    return avg_train_loss, avg_val_loss, loss, mse_loss_preds, mae_loss_preds, mse_loss_10min, mae_loss_10min, mse_loss_offset, mae_loss_offset
