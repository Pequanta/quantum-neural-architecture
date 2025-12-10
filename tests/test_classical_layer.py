import torch
import torch.nn as nn
import unittest
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vqc_fnn.models.classical_layer import ClassicalLayer

class MockQuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        
    def forward(self, x):
        # Mock quantum output: just return squared input or similar
        # Output dimension should match n_qubits (expectation values)
        return torch.sigmoid(x)

class TestClassicalLayer(unittest.TestCase):
    def test_initialization(self):
        input_dim = 10
        n_qubits = 4
        output_dim = 2
        
        layer = ClassicalLayer(input_dim, n_qubits, output_dim)
        
        self.assertEqual(layer.reduction_layer.in_features, input_dim)
        self.assertEqual(layer.reduction_layer.out_features, n_qubits)
        self.assertEqual(layer.post_process[-1].out_features, output_dim)

    def test_fit_pca(self):
        input_dim = 5
        n_qubits = 2
        output_dim = 1
        N = 100
        
        # Create correlated data
        # x1 = rand, x2 = 2*x1, ...
        data = torch.randn(N, input_dim)
        data[:, 1] = 2 * data[:, 0]
        
        layer = ClassicalLayer(input_dim, n_qubits, output_dim)
        
        # Save original weights
        orig_weight = layer.reduction_layer.weight.clone()
        
        layer.fit_pca(data)
        
        # Weights should have changed
        self.assertFalse(torch.allclose(layer.reduction_layer.weight, orig_weight))
        
        # Check dimensions
        self.assertEqual(layer.reduction_layer.weight.shape, (n_qubits, input_dim))

    def test_forward_pass_no_quantum(self):
        input_dim = 10
        n_qubits = 4
        output_dim = 3
        batch_size = 8
        
        layer = ClassicalLayer(input_dim, n_qubits, output_dim)
        x = torch.randn(batch_size, input_dim)
        
        out = layer(x)
        
        self.assertEqual(out.shape, (batch_size, output_dim))

    def test_forward_pass_with_quantum(self):
        input_dim = 10
        n_qubits = 4
        output_dim = 3
        batch_size = 8
        
        layer = ClassicalLayer(input_dim, n_qubits, output_dim)
        q_layer = MockQuantumLayer(n_qubits)
        x = torch.randn(batch_size, input_dim)
        
        out = layer(x, quantum_layer=q_layer)
        
        self.assertEqual(out.shape, (batch_size, output_dim))

    def test_gradient_flow(self):
        input_dim = 10
        n_qubits = 4
        output_dim = 1
        batch_size = 5
        
        layer = ClassicalLayer(input_dim, n_qubits, output_dim)
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))

if __name__ == '__main__':
    unittest.main()
