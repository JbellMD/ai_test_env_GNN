import torch
from models.knowledge_graphs.rgcn import RGCN

def test_rgcn():
    num_entities = 100
    num_relations = 50
    embedding_dim = 32
    num_triples = 1000
    
    model = RGCN(num_entities, num_relations, embedding_dim)
    triples = torch.randint(0, num_entities, (num_triples, 3))
    
    output = model(triples)
    assert output.shape == (num_triples,), \
        f"Expected shape {(num_triples,)}, got {output.shape}"