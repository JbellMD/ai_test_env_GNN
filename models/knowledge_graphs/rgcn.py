import torch
import torch.nn as nn
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        
    def forward(self, triples):
        h = self.entity_emb(triples[:, 0])
        r = self.relation_emb(triples[:, 1])
        t = self.entity_emb(triples[:, 2])
        
        score = torch.sum(h * r * t, dim=1)
        return score