import torch.nn.functional as f
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
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


def calc_loss(preds, target, loss_fn):
    with torch.no_grad():
        losses = []
        for p, t in zip(preds, target):
            p = p.squeeze()
            t = t.squeeze()
            losses.append(loss_fn(p, t).item())
    return sum(losses)/len(losses)


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

class CombinerModel_2(nn.Module):
    def __init__(self, input_size):
        data_type = torch.cuda.FloatTensor
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        return x

model = CombinerModel_2(7)
model.load_state_dict(torch.load("layer2_nochannel_dist.pt"))

loss_fn = nn.MSELoss()
batch_size = 512

_data = pd.read_csv("data/dist-L2-traindata.csv", sep=";")
dev_test_split_index = round(0.9*len(_data))
test_data = _data.loc[dev_test_split_index:]

x_columns = ["pred70", "pred200", "pred700", "ema30", "macd", "rsi5", "volatility100"]
             #"channel_k_min_1200", "channel_k_max_1200", "channel_m_min_1200", "channel_m_max_1200",
             #"channel_k_min_7200", "channel_k_max_7200", "channel_m_min_7200", "channel_m_max_7200"]

test_data_x = torch.tensor(test_data[x_columns].values, dtype=torch.float32)
test_data_y = torch.tensor(test_data[["target"]].values, dtype=torch.float32)
test_data_t = TensorDataset(test_data_x, test_data_y)
test_data_loader = DataLoader(test_data_t, batch_size=batch_size)


print("Testing model\n------------------------------------------")

loss, preds = evaluate_model(test_data_loader, model, loss_fn)
print("\nTest loss: " + "{:0.8f}".format(loss))

preds70 = test_data_x[:, 0]
preds200 = test_data_x[:, 1]
preds700 = test_data_x[:, 2]

print("\nPreds70 loss: {:0.8f}".format(calc_loss(preds70, test_data_y, loss_fn)))
print("Preds200 loss: {:0.8f}".format(calc_loss(preds200, test_data_y, loss_fn)))
print("Preds700 loss: {:0.8f}".format(calc_loss(preds700, test_data_y, loss_fn)))

ema30 = test_data_x[:, 3]
macd = test_data_x[:, 4]
rsi = test_data_x[:, 5]
volatility = test_data_x[:, 6]


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

