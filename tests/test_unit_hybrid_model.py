import unittest
import torch
import numpy as np
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vqc_fnn.models.hybrid_model import HybridVQCFNN

class TestUnitHybridModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 4
        self.num_qubits = 4
        self.output_dim = 3
        self.model = HybridVQCFNN(num_qubits=self.num_qubits, n_layers=2, input_dim=self.input_dim, output_dim=self.output_dim)

    def test_initialization(self):
        self.assertIsInstance(self.model, HybridVQCFNN)
        self.assertEqual(self.model.classical_layer.reduction_layer.in_features, self.input_dim)
        self.assertEqual(self.model.classical_layer.reduction_layer.out_features, self.num_qubits)

    def test_forward_pass_shape(self):
        batch_size = 5
        x = torch.randn(batch_size, self.input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (batch_size, self.output_dim))

    def test_pca_fitting(self):
        # Create dummy data
        data = torch.randn(20, self.input_dim)
        # Save initial weights
        initial_weights = self.model.classical_layer.reduction_layer.weight.clone()
        
        # Fit PCA
        self.model.classical_layer.fit_pca(data)
        
        # Check if weights changed
        self.assertFalse(torch.allclose(self.model.classical_layer.reduction_layer.weight, initial_weights))

if __name__ == '__main__':
    unittest.main()
