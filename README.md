# Graph Neural Networks Environment

A comprehensive environment for implementing and testing various graph neural network techniques, including self-modifying architectures.

## Features

- **Core GNN Models:**
  - Node Classification (GCN)
  - Link Prediction (GAT)
  - Graph Classification (GIN)
  - Knowledge Graphs (RGCN)

- **Self-Modifying Architecture Components:**
  - Architecture Space: Defines possible modifications
  - Modification Controller: Proposes and manages changes
  - Evolution Manager: Handles mutation and crossover
  - Performance Monitor: Tracks model metrics
  - SelfModifyingModel: Base class for self-modifying models

- **Utilities:**
  - Graph visualization
  - Data splitting
  - Model evaluation
  - Architecture tracking

- **Testing:**
  - Unit tests for all model types
  - Test coverage for core functionality
  - Self-modification tests

- **Demonstrations:**
  - Jupyter notebooks with example implementations
  - Sample datasets and visualization
  - Self-modification examples

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai_test_env_GNN.git
   cd ai_test_env_GNN
Set up the environment:
bash
CopyInsert
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Run the tests:
bash
CopyInsert in Terminal
pytest tests/
Usage
Running the Notebooks
Start Jupyter Notebook:
bash
CopyInsert in Terminal
jupyter notebook
Open and run the demonstration notebooks:
notebooks/node_classification_demo.ipynb
notebooks/link_prediction_demo.ipynb
notebooks/graph_classification_demo.ipynb
notebooks/knowledge_graphs_demo.ipynb
notebooks/self_modifying_demo.ipynb
Using Self-Modifying Models
python
CopyInsert
from models.self_modifying.self_modifying_model import SelfModifyingModel
from models.self_modifying.modification_controller import ModificationController

# Initialize components
base_model = ...  # Your base GNN model
model = SelfModifyingModel(base_model)
controller = ModificationController()

# Training loop with self-modification
for epoch in range(100):
    # Train model
    ...
    
    # Evaluate performance and modify if needed
    if controller.should_modify(performance_metrics):
        modification = controller.propose_modification()
        model.modify_architecture(modification)
Directory Structure
CopyInsert
ai_test_env_GNN/
├── models/
│   ├── node_classification/  # GCN implementation
│   ├── link_prediction/      # GAT implementation
│   ├── graph_classification/ # GIN implementation
│   ├── knowledge_graphs/     # RGCN implementation
│   └── self_modifying/       # Self-modifying components
├── datasets/
│   ├── loaders/              # Dataset loading utilities
├── tests/                    # Unit tests
├── utils/                    # Utility functions
├── notebooks/                # Demonstration notebooks
├── requirements.txt          # Python dependencies
└── Dockerfile                # Container configuration
```