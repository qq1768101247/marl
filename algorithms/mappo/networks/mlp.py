import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation()]
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)    