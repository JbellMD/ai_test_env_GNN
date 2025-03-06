import torch
import torch.nn as nn
import torch.nn.functional as F

class GIN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            
        self.layers.append(nn.Linear(hidden_features, out_features))
    
    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.matmul(adj, x)
            x = F.relu(x)
        
        return self.layers[-1](x)