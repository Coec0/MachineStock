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
chunksize = 500000
data_split_ratio=0.8

#input_size= 200
#batch_size= 512
#nbr_epochs= 20
#lr = 0.0001
#save_model_epochs = [5, 10, 20, 100, 500]
#model_name = "models/Swedbank"
#y_column = "5s"
#files_x = ["../python-docker/Swedbank_A/x_Swedbank_A_200_p_channels.csv",]
#files_y = ["../python-docker/Swedbank_A/5_y_Swedbank_A_200.csv",]


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
        n_correct = 0
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

# In[52]:

def train_chunk(model, loss_fn, optimizer, epochrange, x_data, y_data, data_split_ratio, batch_size, filepath):
    train_data, dev_data = splitData(x_data, y_data, data_split_ratio)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    dev_data_loader = DataLoader(dev_data, batch_size=batch_size, drop_last=True)
    return train_model(model, train_data_loader, dev_data_loader, loss_fn, optimizer, epochrange, batch_size, filepath)

def from_norm(x, min_x, max_x):
    return (x*(max_x-min_x+0.0000001)+min_x)

#Start training
def train(files_x, files_y, model, input_size, window_size, loss_fn, optimizer, filepath, epochrange, batch_size, cols_x, col_y, min, max):
    model = model.to(device)
    test_data_x = pd.DataFrame()
    test_data_y = pd.DataFrame()
    last_epoch_train_loss = []
    last_epoch_val_loss = []
    last_epoch_r2 = []
    for i in range(len(files_x)):
        print("Current file: " + files_x[i])
        total_rows = sum(1 for row in open(files_x[i], 'r'))
        number_of_loops = int(total_rows/chunksize)
        print("Number of chunks: " + str(number_of_loops))
        current_loop = 0
        with pd.read_csv(files_x[i], sep=";", dtype="float32", usecols = cols_x, chunksize=chunksize) as reader_x, pd.read_csv(files_y[i], sep=";", dtype="float32", converters = {'ts': int}, chunksize=chunksize, usecols=[col_y]) as reader_y:
            for chunk_x, chunk_y in zip(reader_x, reader_y):
                print("Progress: " + "{:.2f}".format(100 * current_loop/number_of_loops) + "%")
                if(current_loop < data_split_ratio * number_of_loops):
                    train_avg_loss, dev_avg_loss, r2 = train_chunk(model, loss_fn, optimizer, epochrange, chunk_x, chunk_y, data_split_ratio, batch_size,filepath)
                    last_epoch_train_loss.append(train_avg_loss)
                    last_epoch_val_loss.append(dev_avg_loss)
                    last_epoch_r2.append(r2)
                else:
                    print("Append test data")
                    test_data_x = test_data_x.append(chunk_x)
                    test_data_y = test_data_y.append(chunk_y)
                current_loop+=1
                del chunk_x
                del chunk_y

    torch.save(model, filepath+"model.pt")

    test_data_x = torch.tensor(test_data_x.values, dtype=torch.float32)
    test_data_y = torch.tensor(test_data_y.values, dtype=torch.float32)
    test_data = TensorDataset(test_data_x, test_data_y)

    x_avg = test_data_x[:, 0:window_size]
    x_avg = torch.mean(x_avg, 1)

    test_data_loader = DataLoader(test_data, batch_size=2)
    loss, preds, r2 = evaluate_model(test_data_loader, model, loss_fn)

    print(preds[0:50])
    target = [from_norm(t, min, max) for t in test_data_y.flatten().tolist()]
    preds = [from_norm(p, min, max) for p in preds]
    print(preds[0:50])
    x_avg = [from_norm(x, min, max) for x in x_avg.tolist()]

    with io.open(filepath+"log.txt", "a", encoding="utf-8") as f:
        f.write("\n------\nTest loss: " + str(loss))
        f.write("\nTest R^2: " + str(r2))

    plt.plot(list(range(len(preds))), preds, label="Predictions")
    plt.plot(list(range(len(target))), target, label="Target")
    axes = plt.gca()
    axes.set_ylim([160.2,161])
    axes.set_xlim([159000,160000])
    plt.legend()
    plt.savefig(filepath+'zoom.pdf')
    plt.close()

    plt.plot(list(range(len(preds))), preds, label="Predictions")
    plt.plot(list(range(len(target))), target, label="Target")
    plt.plot(list(range(len(x_avg))), x_avg, label="Avg price")
    axes = plt.gca()
    axes.set_ylim([161,163])
    plt.legend()
    axes.set_xlim([100000,120000])
    plt.savefig(filepath+'avg.pdf')
    plt.close()

    plt.plot(list(range(len(preds))), preds, label="Predictions")
    plt.plot(list(range(len(target))), target, label="Target")
    axes = plt.gca()
    plt.legend()
    plt.savefig(filepath+'whole.pdf')
    plt.close()

    avg_train_loss = sum(last_epoch_train_loss)/len(last_epoch_train_loss)
    avg_val_loss = sum(last_epoch_val_loss)/len(last_epoch_val_loss)
    avg_val_r2 = sum(last_epoch_r2)/len(last_epoch_r2)

    return avg_train_loss, avg_val_loss, loss, avg_val_r2, r2
