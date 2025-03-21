import torch
from models.node_classification.gcn import GCN

def test_gcn():
    in_features = 16
    hidden_features = 32
    out_features = 7
    num_nodes = 100
    
    model = GCN(in_features, hidden_features, out_features)
    x = torch.randn(num_nodes, in_features)
    adj = torch.ones(num_nodes, num_nodes)
    
    output = model(x, adj)
    assert output.shape == (num_nodes, out_features), \
        f"Expected shape {(num_nodes, out_features)}, got {output.shape}"