import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, input_shape, hidden_units, output_shape):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_shape, hidden_units[0])
        self.layer2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.layer3 = nn.Linear(hidden_units[1], output_shape)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        return out
