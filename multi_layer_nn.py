import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2304, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)
def custNet(num_first_layer_units,num_hidden_units):
    class myNet(nn.Module):
        def __init__(self):
            super(myNet, self).__init__()
            self.fc1 = nn.Linear(2304, num_first_layer_units)
            self.fc2 = nn.Linear(num_first_layer_units, num_hidden_units)
            self.fc3 = nn.Linear(num_hidden_units, 7)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.softmax(x)
    return myNet
        