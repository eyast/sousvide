import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SousVideDataset(Dataset):
    '''
    Pytorch dataset class that takes a sous vide CSV as an instance
    '''

    def __init__(self, path=None, transform=None, recurse = False, target_distance=20, testing=False, MAVALUE=15):
        self.transform = transform
        if not path and recurse == False:
            folder = "../logs/fillet"
            file = (os.listdir(folder)[0])
            self.path = folder + "/" + file
            self.data = pd.read_csv(self.path)
        elif not path and recurse == True:
            folder = "../logs/"
            ext = ".csv"
            list_of_files = []
            for dirpath, dirname, files in os.walk(folder):
                for name in files:
                    if name == "Fillet_Mignon_2a__KP.1.1_KI.0.01_KD.100_Cycles.1_TargetTemp.56_.csv":
                        continue
                    if name.lower().endswith(ext):
                        list_of_files.append(os.path.join(folder, dirpath, name))
            for i, file in enumerate(list_of_files):
                if i == 0:
                    self.data = pd.read_csv(file)
                else:
                    temp_df = pd.read_csv(file,)
                    self.data = self.data.append(temp_df)
        else:
            self.path = path
            self.data = pd.read_csv(self.path)
        if testing:
            self.data = pd.read_csv("../logs/fillet2/Fillet_Mignon_2a__KP.1.1_KI.0.01_KD.100_Cycles.1_TargetTemp.56_.csv")
        self.data.drop(["actiontime", "Pval", "Ival", "Dval", "movement", "target_temp", "stepcount"],
                        axis=1, inplace=True)
        self.data_diff = self.data.copy()
        self.data_diff["SMA"] = self.data_diff["current_temp"].ewm(span= MAVALUE).mean()
        self.data_diff["target"] = self.data_diff["current_temp"].shift(-target_distance)
        self.data_diff.dropna(inplace=True)
        self.data_diff = self.data_diff[['outcome', 'current_temp', 'SMA', 'target']]
        data_matrix = self.data_diff.values
        data_matrix = data_matrix.astype(np.float)
        self.data = StandardScaler().fit_transform(data_matrix)
        self.data = data_matrix[:,:-1]
        print(self.data.shape)
        self.data = torch.from_numpy(self.data)
        self.target = data_matrix[:,-1]
        print(self.target.shape)
        self.target = np.reshape(self.target, (-1, 1))
        print(self.target.shape)
        self.target = torch.from_numpy(self.target)

    def scale(self, x):
        x = np.reshape(x, (-1, 1))
        self.ss = StandardScaler()
        x = self.ss.fit_transform(x)
        x = np.reshape(x, (-1, 1))
        return x

    def _width(self):
        return self.data.shape[1]

    def __len__(self):
        return (self.data_diff.shape)[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        data = self.data[index]
        target = self.target[index]
        if self.transform:
            data = data.unsqueeze(dim=0)
            data = data.float()           
            target = target.unsqueeze(dim=0)
            target = target.float()     
        return data, target

class MyLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super (MyLSTM, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True).to(self.device)
        self.fc_1 = nn.Linear(hidden_size, 128).to(self.device)
        self.fc = nn.Linear(128, num_classes).to(self.device)
        self.drop = nn.Dropout(0.2, inplace=True)
        self.relu = nn.ReLU()
        
        print(self.device)
        self.to(self.device)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.drop(out)
        return out


def train(model, dataloader, validation_dataloader=False, loss_fn=nn.L1Loss(), epochs=100, optimizer=None):
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = 0
    vall_loss = 0
    for epoch in range(epochs):
        for data in dataloader:
            x, y = data
            x = x.to(model.device)
            y = y.to(model.device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if validation_dataloader:
            for x_val, y_val in validation_dataloader:
                x_val = x_val.to(model.device)
                y_val = y_val.to(model.device)
                out_val = model(x_val)
                minibatch_loss = loss_fn(out_val, y_val)
                vall_loss += minibatch_loss.item()
        if not validation_dataloader:
            print(f"epoch: {epoch}, loss: {train_loss / 100}")
        else:
            print(f"epoch: {epoch}, training loss: {train_loss / 100}, validation loss = {vall_loss/ 100}")
        train_loss = 0
        val_loss = 0

