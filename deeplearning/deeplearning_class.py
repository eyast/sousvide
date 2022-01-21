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

    def __init__(self, path=None, transform=None, recurse = False, target_distance=20):
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
        self.data.drop(["actiontime", "Pval", "Ival", "Dval", "movement", "target_temp", "stepcount"],
                        axis=1, inplace=True)
        self.data_diff = self.data.copy()
        self.data_diff["target"] = self.data_diff["current_temp"].shift(-target_distance)
        self.data_diff.dropna(inplace=True)
        self.ss = StandardScaler()
        data_matrix = self.data_diff.values
        data_matrix - data_matrix.astype(np.float)
        data_matrix = torch.from_numpy(data_matrix)
        self.data = data_matrix[:,:-1]
        self.data = self.ss.fit_transform(self.data)
        self.data = torch.from_numpy(self.data)
        self.target = data_matrix[:,-1]
        self.target = np.reshape(self.target, (-1, 1))
        self.target = self.ss.fit_transform(self.target)
        self.target = torch.from_numpy(self.target)

        

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
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc_1 = nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


def train(model, dataloader, loss_fn=None, epochs=None, optimizer=None):
    if not loss_fn:
        loss_fn = nn.L1Loss()
    if not epochs:
        epochs= 1000
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = 0
    for epoch in range(epochs):
        for data in dataloader:
            x, y = data
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"epoch: {epoch}, loss: {train_loss / 100}")
        train_loss = 0

