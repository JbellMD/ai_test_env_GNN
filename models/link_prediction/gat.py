import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, heads=8):
        super().__init__()
        self.heads = heads
        
        self.attentions = nn.ModuleList([
            nn.Linear(in_features, hidden_features) 
            for _ in range(heads)
        ])
        
        self.out_att = nn.Linear(hidden_features * heads, out_features)
    
    def forward(self, x, adj):
        heads = [F.elu(att(x)) for att in self.attentions]
        x = torch.cat(heads, dim=1)
        x = torch.matmul(adj, x)
        return self.out_att(x)