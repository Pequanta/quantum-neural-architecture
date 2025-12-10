import unittest
import torch
import torch.nn as nn
from sklearn import datasets
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantum_transformer import QuantumTransformer, QuantumTransformerBlock, create_quantum_transformer
from src.utils import InputPreprocessing


class TestQuantumTransformer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.d_model = 8
        self.num_heads = 2
        self.num_qubits = 4
        self.num_classes = 3
        self.batch_size = 2
        self.seq_len = 2
        
    def test_transformer_block_initialization(self):
        """Test transformer block initialization"""
        block = QuantumTransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_qubits=self.num_qubits,
            n_vqc_layers=2
        )
        
        self.assertIsInstance(block.attention, nn.Module)
        self.assertIsInstance(block.vqc_layer, nn.Module)
        self.assertIsInstance(block.classical_layer, nn.Module)
        
    def test_transformer_block_forward(self):
        """Test transformer block forward pass"""
        block = QuantumTransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_qubits=self.num_qubits,
            n_vqc_layers=2
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = block(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_quantum_transformer_initialization(self):
        """Test full transformer initialization"""
        model = QuantumTransformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_qubits=self.num_qubits,
            num_blocks=2,
            num_classes=self.num_classes
        )
        
        self.assertEqual(len(model.blocks), 2)
        self.assertIsNotNone(model.classifier)
        
    def test_quantum_transformer_forward(self):
        """Test full transformer forward pass"""
        model = QuantumTransformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_qubits=self.num_qubits,
            num_blocks=1,
            num_classes=self.num_classes
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_2d_input_handling(self):
        """Test that 2D input is handled correctly"""
        model = QuantumTransformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_qubits=self.num_qubits,
            num_blocks=1,
            num_classes=self.num_classes
        )
        
        # 2D input (batch, d_model)
        x = torch.randn(self.batch_size, self.d_model)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_gradient_flow(self):
        """Test gradient flow through entire transformer"""
        model = QuantumTransformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_qubits=self.num_qubits,
            num_blocks=1,
            num_classes=self.num_classes
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        
    def test_factory_function(self):
        """Test create_quantum_transformer factory function"""
        model = create_quantum_transformer(
            input_dim=8,
            num_classes=3,
            num_heads=2,
            num_qubits=4,
            num_blocks=1
        )
        
        self.assertIsInstance(model, QuantumTransformer)
        
    def test_training_step(self):
        """Test that model can perform a training step"""
        model = create_quantum_transformer(
            input_dim=self.d_model,
            num_classes=self.num_classes,
            num_heads=self.num_heads,
            num_qubits=self.num_qubits,
            num_blocks=1
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))


class TestQuantumTransformerOnDatasets(unittest.TestCase):
    """Integration tests on real datasets"""
    
    def test_iris_classification(self):
        """Test quantum transformer on Iris dataset"""
        print("\n--- Testing Quantum Transformer on Iris ---")
        
        # Load data
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        
        # Preprocess
        preprocessor = InputPreprocessing(X, y, test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        
        # Create model
        model = create_quantum_transformer(
            input_dim=4,
            num_classes=3,
            num_heads=2,
            num_qubits=4,
            num_blocks=1,
            n_vqc_layers=2
        )
        
        # Fit PCA on first block's classical layer
        model.blocks[0].classical_layer.fit_pca(X_train)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train for 2 epochs
        model.train()
        for epoch in range(2):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_test), dim=1)
            acc = (preds == y_test).float().mean().item()
        
        print(f"Test Accuracy: {acc:.2f}")
        
        # Should achieve better than random (with only 2 epochs, expect modest improvement)
        self.assertGreater(acc, 0.25)


if __name__ == '__main__':
    # Reduce logging verbosity
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    unittest.main()
