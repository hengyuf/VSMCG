import torch
import torch.nn as nn
class VIScaler(nn.Module):
    def __init__(self, hidden_size=16,device="cpu"):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size).to(device)
        self.fc12= nn.Linear(hidden_size, hidden_size).to(device)
        self.fc13= nn.Linear(hidden_size, hidden_size).to(device)


        self.fc21 = nn.Linear(hidden_size, 1).to(device)
        self.fc22 = nn.Linear(hidden_size, 1).to(device)
        self.fc23 = nn.Linear(hidden_size, 1).to(device)
        self.fc24 = nn.Linear(hidden_size, 1).to(device)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc12(x))
        x= torch.relu(self.fc13(x))
        x1 = self.fc21(x)
        x2 = self.fc22(x)
        x3 = self.fc23(x)
        x4 = self.fc24(x)
        x1 = 1e-2+torch.sigmoid(x1/10)*10
        x2 = 1e-4+torch.sigmoid(x2)
        x3 = 1e-4+torch.sigmoid(x3)
        x4 = 1e-4+torch.sigmoid(x4)
        return x1,x2,x3,x4
