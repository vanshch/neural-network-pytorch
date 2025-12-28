from sklearn.model_selection import train_test_split
from typing import Tuple,Any
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

URL = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
EPOCHS = 100

class Model(nn.Module):
    def __init__(self, inp: int = 4, hid_layer1: int = 8, hid_layer2: int = 9, output: int = 3, random_seed: int = 41) -> None:
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(inp, hid_layer1)
        self.fc2: nn.Linear = nn.Linear(hid_layer1, hid_layer2)
        self.out: nn.Linear = nn.Linear(hid_layer2, output)
        # setting up the seed for reproducibility
        torch.manual_seed(41)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.out(X)
        return X

class ELT:
    def __init__(self):
        pass
    def extract(self,url:str) -> pd.DataFrame:
        return pd.read_csv(url)

    def transform(self, df:pd.DataFrame,column:str = "species")->Tuple[np.ndarray,np.ndarray]:
        df[column] = df[column].astype("category").cat.codes
        X = df.drop(column,axis = 1).values
        y = df[column].values
        return X,y

    def load(self,x:np.ndarray,y:np.ndarray ,test_size:float = 0.2, random_state:int = 41) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = test_size,random_state = random_state)
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)

        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        return X_train,X_test,y_train,y_test

def main():
    model = Model()
    elt = ELT()
    df = elt.extract(URL)
    X,y = elt.transform(df)
    X_train,X_test,y_train,y_test = elt.load(X,y)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    losses = []

    for i in range(EPOCHS):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred,y_train)
        losses.append(loss.detach().numpy())

        if i % 10 == 0:
            print(f'epoch {i} and loss {loss}')

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()