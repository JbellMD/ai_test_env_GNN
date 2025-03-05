import torch
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(edge_index, num_nodes):
    """Visualize a graph using NetworkX"""
    G = nx.Graph()
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, node_size=50, width=0.5)
    plt.show()

def split_graph_data(data, train_ratio=0.8):
    """Split graph data into training and test sets"""
    num_nodes = data.num_nodes
    num_train = int(num_nodes * train_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[num_train:] = True
    
    return train_mask, test_mask