from torch_geometric.datasets import FacebookPagePage

def load_social_graph():
    """Load social graph dataset"""
    dataset = FacebookPagePage(root='data/SocialGraphs')
    return dataset