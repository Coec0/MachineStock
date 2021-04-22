import torch.nn.functional as f
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
import numpy as np
import pandas as pd

#Params
dtype = torch.cuda.FloatTensor
device = torch.device('cuda:0')


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
    return avg_loss, predictions


def train_model(model, train_data_loader, dev_data_loader, loss_fn, optimizer, epochrange):
    train_avg_loss = 0
    dev_avg_loss = 0
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
        dev_avg_loss, _ = evaluate_model(dev_data_loader, model, loss_fn)

        # Display metrics
        display_str = 'Epoch {} '
        display_str += '\tLoss: {:.6f} '
        display_str += '\tLoss (val): {:.6f}'
        print(display_str.format(epoch, train_avg_loss, dev_avg_loss))


class CombinerModel(nn.Module):
    def __init__(self, input_size):
        data_type = torch.cuda.FloatTensor
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size*2).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2 = nn.Linear(input_size*2, round(input_size*0.5)).type(data_type)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3 = nn.Linear(round(input_size*0.5), 1).type(data_type)
        self.fc3.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))
        y = f.leaky_relu(self.fc3(x))
        return y


model = CombinerModel(3)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch_range = 20
batch_size = 128

_data = pd.read_csv("data/dist-L2-traindata.csv", sep=";")
split_index = round(0.8*len(_data))
train_data = _data.loc[:split_index]
eval_data = _data.loc[split_index:]

print(train_data.head())

train_data_x = torch.tensor(train_data[["pred70", "pred200", "pred700"]].values, dtype=torch.float32)
train_data_y = torch.tensor(train_data[["target"]].values, dtype=torch.float32)
train_data_t = TensorDataset(train_data_x, train_data_y)
train_data_loader = DataLoader(train_data_t, batch_size=batch_size, drop_last=True)

eval_data_x = torch.tensor(eval_data[["pred70", "pred200", "pred700"]].values, dtype=torch.float32)
eval_data_y = torch.tensor(eval_data[["target"]].values, dtype=torch.float32)
eval_data_t = TensorDataset(eval_data_x, eval_data_y)
eval_data_loader = DataLoader(eval_data_t, batch_size=batch_size, drop_last=True)

print("Starting training\n------------------------------------------\n")
train_model(model, train_data_loader, eval_data_loader, loss_fn, optimizer, epoch_range)

torch.save(model.state_dict(), "layer2_model_dist.pt")
