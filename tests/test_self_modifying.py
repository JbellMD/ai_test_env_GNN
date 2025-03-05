import torch
from models.self_modifying.architecture_space import ArchitectureSpace
from models.self_modifying.modification_controller import ModificationController

def test_architecture_space():
    space = ArchitectureSpace()
    assert 'add_layer' in space.operations
    assert 'conv' in space.layer_types

def test_modification_controller():
    controller = ModificationController()
    loss_history = [0.5] * 20
    assert controller.evaluate_stability(None, loss_history) == True