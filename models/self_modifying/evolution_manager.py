import random
from typing import List, Dict, Any
import torch
import torch.nn as nn

class EvolutionManager:
    def __init__(self, mutation_rate=0.1, crossover_rate=0.5):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def mutate(self, model: nn.Module) -> nn.Module:
        """Apply mutations to the model architecture"""
        for name, param in model.named_parameters():
            if random.random() < self.mutation_rate:
                # Apply mutation to parameter
                noise = torch.randn_like(param) * 0.1
                param.data.add_(noise)
        return model
    
    def crossover(self, model1: nn.Module, model2: nn.Module) -> nn.Module:
        """Combine features from two models"""
        new_model = type(model1)()
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), 
                                                  model2.named_parameters()):
            if random.random() < self.crossover_rate:
                # Use parameter from model2
                getattr(new_model, name1).data = param2.data
            else:
                # Use parameter from model1
                getattr(new_model, name1).data = param1.data
        return new_model