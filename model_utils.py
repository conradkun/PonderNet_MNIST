from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = self.droput(x)
        x = F.relu(self.h2o(x))
        return x


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = self.droput(x)
        x = F.relu(self.h2o(x))
        return x
