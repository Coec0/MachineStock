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

class CombinerModel_2(nn.Module):
    def __init__(self, input_size):
        data_type = torch.cuda.FloatTensor
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1).type(data_type)
        self.fc1.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x):
        x = f.leaky_relu(self.fc1(x))
        return x


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


model = CombinerModel_3(4)#CombinerModel_3_['ema15', 'macd', 'rsi5', 'rsi30', 'volatility100', 'volatility50']_lr0.001_NordeaPred70
model.load_state_dict(torch.load("data/auto/models/CombinerModel_3_volatility50_lr0.01_None._state_dict.pt"))

loss_fn = nn.MSELoss()
batch_size = 512
#SwedbankPred70;SwedbankPred200;SwedbankPred700;NordeaPred70;NordeaPred200;NordeaPred700;ema30;ema15;macd;rsi5;rsi30;volatility100;volatility50;target;avg;stdev;ts
_data = pd.read_csv("data/Swedbank_A_dist2_train_zscore_Nordea.csv", sep=";")
dev_test_split_index = round(0.9*len(_data))
test_data = _data.loc[dev_test_split_index:]

x_columns = ["SwedbankPred70","SwedbankPred200","SwedbankPred700", "volatility50"] # "ema30", "macd", "rsi5",
             #"channel_k_min_1200", "channel_k_max_1200", "channel_m_min_1200", "channel_m_max_1200",
             #"channel_k_min_7200", "channel_k_max_7200", "channel_m_min_7200", "channel_m_max_7200"]

test_data_x = torch.tensor(test_data[x_columns].values, dtype=torch.float32)
test_data_y = torch.tensor(test_data[["target"]].values, dtype=torch.float32)
test_data_t = TensorDataset(test_data_x, test_data_y)
test_data_loader = DataLoader(test_data_t, batch_size=batch_size)


print("Testing model\n------------------------------------------")

loss, preds = evaluate_model(test_data_loader, model, loss_fn)
#print("\nTest loss: " + "{:0.8f}".format(loss))

test_data_y_ = test_data[["target", "avg", "stdev"]]
preds70,_ = denorm(test_data_x[:, 0].numpy(), test_data_y_)
preds200,_ = denorm(test_data_x[:, 1].numpy(), test_data_y_)
preds700,_ = denorm(test_data_x[:, 2].numpy(), test_data_y_)

preds, target = denorm(preds, test_data_y_)

mse_loss_preds, mae_loss_preds, mse_loss_10min, mae_loss_10min, mse_loss_offset, mae_loss_offset = calc_losses(preds, target, test_data["avg"])
mse_loss_preds70, mae_loss_preds70, mse_loss_10min70, mae_loss_10min70, mse_loss_offset70, mae_loss_offset70 = calc_losses(preds70, target, test_data["avg"])
mse_loss_preds200, mae_loss_preds200, mse_loss_10min200, mae_loss_10min200, mse_loss_offset200, mae_loss_offset200 = calc_losses(preds200, target, test_data["avg"])
mse_loss_preds700, mae_loss_preds700, mse_loss_10min700, mae_loss_10min700, mse_loss_offset700, mae_loss_offset700 = calc_losses(preds700, target, test_data["avg"])

row ="{:.5f}".format(mse_loss_preds) + ";" + "{:.5f}".format(
    mae_loss_preds) + ";" + "{:.5f}".format(mse_loss_preds70) + ";" + "{:.5f}".format(
    mae_loss_preds70) + ";" + "{:.5f}".format(mse_loss_preds200) + ";" + "{:.5f}".format(mae_loss_preds200)  + ";" + "{:.5f}".format(mse_loss_preds700)+ ";" + "{:.5f}".format(mae_loss_preds700)

print(row)
# Plot test predictions
plt.plot(list(range(len(preds))), preds, label="Predictions")
plt.plot(list(range(len(preds))), target, label="Target")
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

