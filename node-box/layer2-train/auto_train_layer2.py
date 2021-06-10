import io
import torch.nn.functional as f
import torch
from torch import nn

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


def train_model(model, train_data_loader, dev_data_loader, loss_fn, optimizer, epochrange, logfile):
    best_eval_loss = 1000000000
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
        s = display_str.format(epoch, train_avg_loss, dev_avg_loss)
        with io.open(logfile, "a", encoding="utf-8") as f:
            f.write(s+"\n")

        if dev_avg_loss > best_eval_loss+0.0001:
            break
        elif dev_avg_loss < best_eval_loss:
            best_eval_loss = dev_avg_loss

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


class LinReg(nn.Module):
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
