import numpy as np
from typing import List, Dict, Any

class PerformanceMonitor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.metrics_history = []
        
    def track_metrics(self, metrics: Dict[str, Any]) -> None:
        """Track model performance metrics"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
    
    def get_performance_trend(self) -> Dict[str, float]:
        """Calculate performance trends"""
        if not self.metrics_history:
            return {}
            
        trends = {}
        for metric in self.metrics_history[0].keys():
            values = [m[metric] for m in self.metrics_history]
            trends[metric] = np.polyfit(range(len(values)), values, 1)[0]
        return trends