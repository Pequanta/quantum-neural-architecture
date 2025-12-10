import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from pennylane import numpy as np
import unittest
import sys
import os

# Adjust path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention_block import QuantumAttention
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToySequenceDataset(Dataset):
    def __init__(self, num_samples, seq_len, d_model):
        self.inputs = torch.randn(num_samples, seq_len, d_model)
        self.labels = (self.inputs.sum(dim=(1,2)) > 0).float().unsqueeze(1)  # Binary label: positive sum parity

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Training setup
def train_model(model, train_loader, epochs=10, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs).mean(dim=(1,2)).unsqueeze(1)  # Pool attention outputs for classification
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")  # Keep console print for compatibility

# Testing setup
def test_model(model, test_loader):
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs).mean(dim=(1,2)).unsqueeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            logger.info(f"Test Batch {batch_idx+1}: Loss = {loss.item():.4f}")
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    logger.info(f"Test Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy

# Unit tests
class TestQuantumAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 4
        self.seq_len = 2  # Small for testing
        self.model = QuantumAttention(self.d_model)
        self.batch_size = 2

    def test_forward_shape(self):
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model)
        outputs = self.model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_training_step(self):
        inputs = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        outputs = self.model(inputs)
        loss = outputs.sum()
        loss.backward()
        self.assertIsNotNone(self.model.q_params.grad)  # Check gradients flow

# Example usage
if __name__ == "__main__":
    # Dataset and loaders
    train_dataset = ToySequenceDataset(100, seq_len=2, d_model=4)
    test_dataset = ToySequenceDataset(20, seq_len=2, d_model=4)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    # Model
    model = QuantumAttention(d_model=4)

    # Train
    train_model(model, train_loader, epochs=5)

    # Test
    test_model(model, test_loader)

    # Run unit tests
    unittest.main(exit=False)
