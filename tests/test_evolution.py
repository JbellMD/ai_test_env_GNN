import torch
import torch.nn as nn
from models.self_modifying.evolution_manager import EvolutionManager

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

def test_mutation():
    model = TestModel()
    manager = EvolutionManager(mutation_rate=1.0)
    original_params = model.fc.weight.data.clone()
    mutated_model = manager.mutate(model)
    assert not torch.equal(original_params, mutated_model.fc.weight.data)

def test_crossover():
    model1 = TestModel()
    model2 = TestModel()
    manager = EvolutionManager(crossover_rate=1.0)
    crossed_model = manager.crossover(model1, model2)
    assert torch.equal(crossed_model.fc.weight.data, model2.fc.weight.data)