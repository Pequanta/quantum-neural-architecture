import unittest
import torch
import torch.nn as nn
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention_block.q_multihead_attention import QuantumMultiheadAttention

class TestMultiheadAttentionIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.d_model = 8
        self.num_heads = 2
        self.n_qubits = 4
        self.batch_size = 1  # Small for integration tests
        self.seq_len = 2
        
    def test_basic_forward_pass(self):
        """Test basic forward pass with small input"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = model(inputs)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check no NaN or Inf values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_training_loop(self):
        """Test that model can be trained for a few steps"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Create dummy data
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model)
        targets = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Training steps
        initial_loss = None
        for step in range(2):  # Just 2 steps for testing
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if step == 0:
                initial_loss = loss.item()
        
        # Check that loss changed (model is learning)
        final_loss = loss.item()
        self.assertNotEqual(initial_loss, final_loss)
        
    def test_attention_output_consistency(self):
        """Test that same input produces same output (deterministic)"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        model.eval()
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        with torch.no_grad():
            output1 = model(inputs)
            output2 = model(inputs)
        
        # Should produce same output for same input
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
        
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        model = QuantumMultiheadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            n_qubits=self.n_qubits
        )
        
        for seq_len in [1, 2, 3]:
            inputs = torch.randn(self.batch_size, seq_len, self.d_model)
            output = model(inputs)
            self.assertEqual(output.shape, (self.batch_size, seq_len, self.d_model))

if __name__ == '__main__':
    # Reduce logging verbosity
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    unittest.main()
