import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.input_encoder import InputEncoder
from src.vqc_fnn.models.vqc_layer import VQCLayer
from src.vqc_fnn.models.classical_layer import ClassicalLayer

class HybridVQCFNN(nn.Module):
    def __init__(self, num_qubits=4, n_layers=2, input_dim=4, output_dim=3):
        super().__init__()
        self.encoder = InputEncoder(np.zeros(num_qubits), n_qubits=num_qubits)
        self.q_layer = VQCLayer(num_qubits=num_qubits, n_layers=n_layers, encoder=self.encoder)
        
        self.classical_layer = ClassicalLayer(input_dim=input_dim, n_qubits=num_qubits, output_dim=output_dim)

    def forward(self, x):
        # ClassicalLayer handles the full flow: Reduction -> Quantum (optional) -> Post-process
        return self.classical_layer(x, quantum_layer=self.q_layer)
