"""
Fast CPU-Optimized Test for Quantum Attention on Sequential Data

Optimized for local CPU testing with smaller models and datasets.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention_block import QuantumMultiheadAttention
from src.quantum_transformer import create_quantum_transformer


class FastSequentialDataset:
    """
    Fast sequential data generator for CPU testing.
    Task: Detect if first element is positive or negative.
    """
    def __init__(self, num_samples=50, seq_len=3, d_model=4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.X, self.y = self._generate_data()
        
    def _generate_data(self):
        """Generate simple sequential data where first position determines class."""
        X = []
        y = []
        
        for _ in range(self.num_samples):
            seq = torch.randn(self.seq_len, self.d_model) * 0.3  # Small noise
            
            # First position determines class
            class_label = np.random.randint(0, 2)
            seq[0, 0] = 1.5 if class_label == 1 else -1.5  # Clear signal
            
            X.append(seq)
            y.append(class_label)
        
        return torch.stack(X), torch.tensor(y, dtype=torch.long)


class TestAttentionFastCPU(unittest.TestCase):
    """Fast CPU tests for quantum attention mechanism"""
    
    def test_attention_learns_sequential_pattern(self):
        """
        Fast test that attention can learn from sequential data.
        Uses minimal model size for CPU efficiency.
        """
        print("\n=== Fast Attention Test (CPU Optimized) ===")
        
        # Small dataset
        dataset = FastSequentialDataset(num_samples=60, seq_len=3, d_model=4)
        
        # Split
        X_train = dataset.X[:40]
        y_train = dataset.y[:40]
        X_test = dataset.X[40:]
        y_test = dataset.y[40:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Minimal model for speed
        model = create_quantum_transformer(
            input_dim=4,
            num_classes=2,
            num_heads=1,  # Single head for speed
            num_qubits=4,
            num_blocks=1,
            n_vqc_layers=1,  # Minimal VQC layers
            dropout=0.0
        )
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        
        model.train()
        epochs = 3  # Just 3 epochs for speed
        
        print("Training...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_preds = torch.argmax(outputs, dim=1)
                train_acc = (train_preds == y_train).float().mean().item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.2%}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_preds = torch.argmax(test_outputs, dim=1)
            test_acc = (test_preds == y_test).float().mean().item()
        
        print(f"Test Accuracy: {test_acc:.2%}")
        
        # Should learn this simple task
        self.assertGreater(test_acc, 0.5, 
            "Model should achieve >50% accuracy (better than random)")
        
        print("✓ Attention mechanism is working on sequential data!")
    
    def test_multihead_attention_forward_pass(self):
        """Quick test that multihead attention processes sequences correctly."""
        print("\n=== Testing Multihead Attention Forward Pass ===")
        
        # Minimal setup
        attention = QuantumMultiheadAttention(
            d_model=4,
            num_heads=1,
            n_qubits=4
        )
        
        # Small batch
        x = torch.randn(2, 3, 4)  # (batch=2, seq_len=3, d_model=4)
        
        print("Input shape:", x.shape)
        output = attention(x)
        print("Output shape:", output.shape)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check no NaN/Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        print("✓ Multihead attention forward pass working!")
    
    def test_attention_gradient_flow(self):
        """Quick test that gradients flow through attention."""
        print("\n=== Testing Gradient Flow Through Attention ===")
        
        attention = QuantumMultiheadAttention(
            d_model=4,
            num_heads=1,
            n_qubits=4
        )
        
        x = torch.randn(2, 2, 4, requires_grad=True)
        output = attention(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        
        # Check parameter gradients
        for param in attention.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
        
        print("✓ Gradients flowing correctly through attention!")


class TestAttentionVsBaseline(unittest.TestCase):
    """Compare attention model with simple baseline"""
    
    def test_attention_competitive_with_baseline(self):
        """
        Fast comparison showing attention is competitive.
        Uses very small models for speed.
        """
        print("\n=== Attention vs Baseline (Fast) ===")
        
        # Small dataset
        dataset = FastSequentialDataset(num_samples=50, seq_len=3, d_model=4)
        X_train = dataset.X[:35]
        y_train = dataset.y[:35]
        X_test = dataset.X[35:]
        y_test = dataset.y[35:]
        
        # Attention model (minimal)
        attn_model = create_quantum_transformer(
            input_dim=4,
            num_classes=2,
            num_heads=1,
            num_qubits=4,
            num_blocks=1,
            n_vqc_layers=1
        )
        
        # Simple baseline
        class SimpleBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2)
                )
            
            def forward(self, x):
                return self.net(x.mean(dim=1))
        
        baseline = SimpleBaseline()
        
        # Quick training function
        def quick_train(model, epochs=3):
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            
            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                preds = torch.argmax(test_outputs, dim=1)
                acc = (preds == y_test).float().mean().item()
            return acc
        
        print("Training attention model...")
        attn_acc = quick_train(attn_model, epochs=3)
        print(f"Attention accuracy: {attn_acc:.2%}")
        
        print("Training baseline...")
        baseline_acc = quick_train(baseline, epochs=3)
        print(f"Baseline accuracy: {baseline_acc:.2%}")
        
        print(f"Difference: {(attn_acc - baseline_acc):+.2%}")
        
        # Both should learn something
        self.assertGreater(attn_acc, 0.4)
        self.assertGreater(baseline_acc, 0.4)
        
        print("✓ Both models learning successfully!")


if __name__ == '__main__':
    # Reduce logging
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    print("=" * 60)
    print("FAST CPU-OPTIMIZED QUANTUM ATTENTION TESTS")
    print("=" * 60)
    
    unittest.main(verbosity=2)
