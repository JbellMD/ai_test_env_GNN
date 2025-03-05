from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def load_citation_graph(name='Cora'):
    """Load citation graph dataset"""
    dataset = Planetoid(root='data/CitationGraphs', name=name, transform=NormalizeFeatures())
    return dataset
