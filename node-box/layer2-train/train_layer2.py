import io

import torch.nn.functional as f
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt

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
            ys.extend(y.tolist() if len(y.size()) > 0 else [y])
            predictions.extend(pred.tolist() if len(pred.size()) > 0 else [pred])
        avg_loss = sum(losses)/len(losses)
    return avg_loss, predictions


def train_model(model, train_data_loader, dev_data_loader, loss_fn, optimizer, epochrange):
    for epoch in range(epochrange):
        predictions = []
        losses = []
        model.train()
        for x, y in train_data_loader:
            y = y.type(dtype).squeeze()
            x = x.type(dtype)
            pred = model(x).squeeze()
            loss = loss_fn(pred, y)
            predictions.extend(pred.tolist() if len(pred.size()) > 0 else [pred])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compute accuracy and loss in the entire training set
        train_avg_loss = sum(losses)/len(losses)
        dev_avg_loss, _ = evaluate_model(dev_data_loader, model, loss_fn)

        # Display metrics
        display_str = 'Epoch {} '
        display_str += '\tLoss: {:.8f} '
        display_str += '\tLoss (val): {:.8f}'
        print(display_str.format(epoch, train_avg_loss, dev_avg_loss))
    return predictions


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


class CombinerModel_3(nn.Module):
    def __init__(self, input_size):
        data_type = torch.cuda.FloatTensor
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size*3).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2 = nn.Linear(input_size*3, round(input_size)).type(data_type)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3 = nn.Linear(round(input_size), 1).type(data_type)
        self.fc3.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        x = f.leaky_relu(self.fc2(x))
        y = f.leaky_relu(self.fc3(x))
        return y


class LinReg(nn.Module): #CombinerModel_2
    def __init__(self, input_size):
        data_type = torch.cuda.FloatTensor
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        return x


class CombinerModel_2(nn.Module):
    def __init__(self, input_size):
        data_type = torch.cuda.FloatTensor
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        return x


model = CombinerModel_3(15)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
epoch_range = 100
batch_size = 512

_data = pd.read_csv("data/dist-L2-traindata.csv", sep=";")
train_dev_split_index = round(0.8*len(_data))
dev_test_split_index = round(0.9*len(_data))
train_data = _data.loc[:train_dev_split_index]
eval_data = _data.loc[train_dev_split_index:dev_test_split_index]
test_data = _data.loc[dev_test_split_index:]

print(train_data.head())
x_columns = ["pred70", "pred200", "pred700", "ema30", "macd", "rsi5", "volatility100",
             "channel_k_min_1200", "channel_k_max_1200", "channel_m_min_1200", "channel_m_max_1200",
             "channel_k_min_7200", "channel_k_max_7200", "channel_m_min_7200", "channel_m_max_7200"]

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

print("Starting training\n------------------------------------------\n")
train_preds = train_model(model, train_data_loader, eval_data_loader, loss_fn, optimizer, epoch_range)
torch.save(model.state_dict(), "layer2_model_dict.pt")


print("Testing model\n------------------------------------------\n")

loss, preds = evaluate_model(test_data_loader, model, loss_fn)
print("\nTest loss: " + str(loss))

preds70 = test_data_x[:, 0]
preds200 = test_data_x[:, 1]
preds700 = test_data_x[:, 2]

ema30 = test_data_x[:, 3]
macd = test_data_x[:, 4]
rsi = test_data_x[:, 5]
volatility = test_data_x[:, 6]

# Plot train predictions
plt.plot(list(range(len(train_preds))), train_preds, label="Train Predictions")
plt.plot(list(range(len(train_preds))), train_data_y, label="Train Target")
axes = plt.gca()
plt.legend()
plt.show()

# Plot test predictions
plt.plot(list(range(len(preds))), preds, label="Predictions")
plt.plot(list(range(len(preds))), test_data_y, label="Target")
plt.plot(list(range(len(preds))), preds70, label="Pred70")
plt.plot(list(range(len(preds))), preds200, label="preds200")
plt.plot(list(range(len(preds))), preds700, label="preds700")
#plt.plot(list(range(len(preds))), ema30, label="ema30")
#plt.plot(list(range(len(preds))), macd, label="macd")
#plt.plot(list(range(len(preds))), rsi, label="rsi")
#plt.plot(list(range(len(preds))), volatility, label="volatility")
plt.gca()
plt.legend()
plt.show()

