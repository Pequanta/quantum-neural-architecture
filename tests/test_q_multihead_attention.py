import torch
import torch.nn as nn
import unittest
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention_block import QuantumMultiheadAttention

class TestQuantumMultiheadAttention(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.d_model = 8
        self.num_heads = 2
        self.n_qubits = 4
        self.batch_size = 2
        self.seq_len = 3
        
    def test_initialization(self):
        """Test that the model initializes correctly"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        self.assertEqual(model.d_model, self.d_model)
        self.assertEqual(model.num_heads, self.num_heads)
        self.assertEqual(model.head_dim, self.d_model // self.num_heads)
        self.assertEqual(len(model.heads_q_params), self.num_heads)
        self.assertEqual(len(model.heads_k_params), self.num_heads)
        self.assertEqual(len(model.heads_v_params), self.num_heads)
        self.assertEqual(len(model.heads_cond_params), self.num_heads)
        
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = model(inputs)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.any(inputs.grad != 0))
        
        # Check that all head parameters have gradients
        for head_idx in range(self.num_heads):
            self.assertIsNotNone(model.heads_q_params[head_idx].grad)
            self.assertIsNotNone(model.heads_k_params[head_idx].grad)
            self.assertIsNotNone(model.heads_v_params[head_idx].grad)
            self.assertIsNotNone(model.heads_cond_params[head_idx].grad)
    
    def test_multiple_heads(self):
        """Test with different numbers of heads"""
        for num_heads in [1, 2, 4]:
            d_model = 8  # Must be divisible by num_heads
            model = QuantumMultiheadAttention(
                d_model=d_model,
                num_heads=num_heads,
                n_qubits=self.n_qubits
            )
            
            inputs = torch.randn(self.batch_size, self.seq_len, d_model)
            output = model(inputs)
            
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, d_model))
    
    def test_single_vs_multihead(self):
        """Test that single head multihead attention behaves reasonably"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=1,
            n_qubits=self.n_qubits
        )
        
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = model(inputs)
        
        # Should produce valid output
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_parameter_independence(self):
        """Test that different heads have independent parameters"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        # Check that parameters for different heads are different
        for i in range(self.num_heads - 1):
            self.assertFalse(torch.allclose(
                model.heads_q_params[i], 
                model.heads_q_params[i + 1]
            ))

if __name__ == '__main__':
    # Run tests with reduced verbosity for quantum circuit output
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    unittest.main()
