"""
Quantum Transformer

Integrated quantum transformer architecture combining:
- Quantum Multihead Attention
- VQC Layers
- Classical Processing Layers
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Adjust path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention_block import QuantumMultiheadAttention
from src.vqc_fnn.models import VQCLayer, ClassicalLayer
from src.utils import InputEncoder


class QuantumTransformerBlock(nn.Module):
    """
    Single transformer block with quantum components.
    
    Architecture:
    1. Quantum Multihead Attention
    2. Add & Norm (residual connection)
    3. VQC Layer (quantum feedforward)
    4. Classical Layer (post-processing)
    5. Add & Norm (residual connection)
    """
    def __init__(self, d_model, num_heads, num_qubits, n_vqc_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_qubits = num_qubits
        
        # Quantum Multihead Attention
        self.attention = QuantumMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            n_qubits=num_qubits
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # VQC-based feedforward
        self.encoder = InputEncoder(np.zeros(num_qubits), n_qubits=num_qubits)
        self.vqc_layer = VQCLayer(
            num_qubits=num_qubits,
            n_layers=n_vqc_layers,
            encoder=self.encoder
        )
        
        # Classical processing layer
        self.classical_layer = ClassicalLayer(
            input_dim=d_model,
            n_qubits=num_qubits,
            output_dim=d_model
        )
        
    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Attention block with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # VQC + Classical feedforward with residual connection
        # Reshape for processing: (batch, seq_len, d_model) -> (batch*seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (batch*seq_len, d_model)
        
        # Process through classical layer which internally uses VQC
        ff_out_flat = self.classical_layer(x_flat, quantum_layer=self.vqc_layer)
        
        # Reshape back: (batch*seq_len, d_model) -> (batch, seq_len, d_model)
        ff_out = ff_out_flat.reshape(batch_size, seq_len, d_model)
        
        x = self.norm2(x + self.dropout2(ff_out))
        
        return x


class QuantumTransformer(nn.Module):
    """
    Full Quantum Transformer model.
    
    Combines multiple QuantumTransformerBlocks for deep quantum processing.
    """
    def __init__(self, d_model, num_heads, num_qubits, num_blocks=2, 
                 n_vqc_layers=2, num_classes=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        # Input projection (if needed)
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_qubits=num_qubits,
                n_vqc_layers=n_vqc_layers,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Output head (for classification tasks)
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_classes)
            )
    
    def forward(self, x, return_features=False):
        """
        Forward pass through quantum transformer.
        
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model)
            return_features: If True, return features before classification
        
        Returns:
            output: Classification logits or features
        """
        # Handle 2D input (batch, d_model) -> (batch, 1, d_model)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling over sequence dimension
        features = x.mean(dim=1)  # (batch, d_model)
        
        if return_features:
            return features
        
        # Classification head
        if self.num_classes is not None:
            return self.classifier(features)
        
        return features


def create_quantum_transformer(input_dim, num_classes, num_heads=2, num_qubits=4, 
                               num_blocks=2, n_vqc_layers=2, dropout=0.1):
    """
    Factory function to create a quantum transformer with PCA initialization.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        num_heads: Number of attention heads
        num_qubits: Number of qubits per VQC
        num_blocks: Number of transformer blocks
        n_vqc_layers: Number of layers in each VQC
        dropout: Dropout rate
    
    Returns:
        model: QuantumTransformer instance
    """
    model = QuantumTransformer(
        d_model=input_dim,
        num_heads=num_heads,
        num_qubits=num_qubits,
        num_blocks=num_blocks,
        n_vqc_layers=n_vqc_layers,
        num_classes=num_classes,
        dropout=dropout
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating Quantum Transformer...")
    model = create_quantum_transformer(
        input_dim=8,
        num_classes=3,
        num_heads=2,
        num_qubits=4,
        num_blocks=1,
        n_vqc_layers=2
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 3
    x = torch.randn(batch_size, seq_len, 8)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print("Quantum Transformer created successfully!")
