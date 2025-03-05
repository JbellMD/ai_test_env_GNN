import torch
import numpy as np
from typing import Dict, Any

class ModificationController:
    def __init__(self, stability_threshold=0.1):
        self.stability_threshold = stability_threshold
        self.modification_history = []
        
    def evaluate_stability(self, model: torch.nn.Module, 
                         loss_history: List[float]) -> bool:
        """Evaluate model stability based on loss history"""
        if len(loss_history) < 10:
            return True
            
        recent_loss = np.mean(loss_history[-10:])
        previous_loss = np.mean(loss_history[-20:-10])
        return abs(recent_loss - previous_loss) < self.stability_threshold
    
    def propose_modification(self, model: torch.nn.Module, 
                           performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a modification based on performance metrics"""
        # Implementation of modification proposal
        return {
            'operation': 'add_layer',
            'config': {
                'type': 'linear',
                'params': {
                    'in_features': 128,
                    'out_features': 64
                }
            }
        }