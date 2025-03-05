from torch_geometric.datasets import WordNet18

def load_knowledge_graph():
    """Load knowledge graph dataset"""
    dataset = WordNet18(root='data/KnowledgeGraphs')
    return dataset