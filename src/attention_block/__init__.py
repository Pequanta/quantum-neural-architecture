"""
Attention Block Module

This module contains quantum attention mechanisms.
"""

from .q_singlehead_attention import QuantumAttention
from .q_multihead_attention import QuantumMultiheadAttention

__all__ = [
    'QuantumAttention',
    'QuantumMultiheadAttention',
]
