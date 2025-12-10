import pennylane as qml
import torch
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumMultiheadAttention(nn.Module):
    """
    Quantum Multihead Attention using variational quantum circuits.
    Based on the single-head quantum attention architecture with multiple parallel heads.
    """
    def __init__(self, d_model, num_heads=4, n_qubits=4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_qubits = n_qubits
        self.head_dim = d_model // num_heads
        
        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Parameters for each head (similar to single-head architecture)
        # Each head has: q_params, k_params, v_params, cond_params
        self.heads_q_params = nn.ParameterList([
            nn.Parameter(torch.randn(6)) for _ in range(num_heads)  # 2 wires * 3
        ])
        self.heads_k_params = nn.ParameterList([
            nn.Parameter(torch.randn(6)) for _ in range(num_heads)  # 2 wires * 3
        ])
        self.heads_v_params = nn.ParameterList([
            nn.Parameter(torch.randn(12)) for _ in range(num_heads)  # 4 wires * 3
        ])
        self.heads_cond_params = nn.ParameterList([
            nn.Parameter(torch.randn(9)) for _ in range(num_heads)  # 3 wires * 3
        ])
        
        # Output projection for each head
        self.head_out_linears = nn.ModuleList([
            nn.Linear(n_qubits - 1, self.head_dim) for _ in range(num_heads)
        ])
        
        # Final output projection to combine all heads
        self.out_proj = nn.Linear(d_model, d_model)

    # Variational ansatz (same as single-head)
    @staticmethod
    def variational_layer(params, wires):
        num_wires = len(wires)
        for i in range(num_wires):
            qml.RX(params[i * 3], wires=wires[i])
            qml.RY(params[i * 3 + 1], wires=wires[i])
            qml.RZ(params[i * 3 + 2], wires=wires[i])
        for i in range(num_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Encoding (same as single-head)
    @staticmethod
    def encode_data(x, wires):
        for i, wire in enumerate(wires):
            qml.RY(x[i % len(x)], wires=wire)

    def create_quantum_kernel(self, head_idx):
        """Create quantum kernel for a specific head"""
        @qml.qnode(self.dev, interface='torch')
        def quantum_kernel(q_params, k_params, x_q, x_k):
            QuantumMultiheadAttention.encode_data(x_q, wires=range(self.n_qubits // 2))
            QuantumMultiheadAttention.variational_layer(q_params, wires=range(self.n_qubits // 2))
            
            QuantumMultiheadAttention.encode_data(x_k, wires=range(self.n_qubits // 2, self.n_qubits))
            qml.adjoint(QuantumMultiheadAttention.variational_layer)(k_params, wires=range(self.n_qubits // 2, self.n_qubits))
            
            return qml.probs(wires=range(self.n_qubits))
        return quantum_kernel

    def create_quantum_attention(self, head_idx):
        """Create full quantum attention circuit for a specific head"""
        quantum_kernel = self.create_quantum_kernel(head_idx)
        
        @qml.qnode(self.dev, interface='torch')
        def quantum_attention(q_params, k_params, v_params, cond_params, x_q, x_k, x_v):
            score = quantum_kernel(q_params, k_params, x_q, x_k)[0]
            
            QuantumMultiheadAttention.encode_data(x_v, wires=range(self.n_qubits))
            QuantumMultiheadAttention.variational_layer(v_params, wires=range(self.n_qubits))
            
            ancilla = self.n_qubits - 1
            qml.RY(score * torch.pi, wires=ancilla)
            qml.ctrl(QuantumMultiheadAttention.variational_layer, control=ancilla)(cond_params, wires=range(self.n_qubits - 1))
            
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits - 1)]
        return quantum_attention

    def process_head(self, head_idx, inputs):
        """Process inputs through a single attention head"""
        batch_size, seq_len, _ = inputs.shape
        
        # Get parameters for this head
        q_params = self.heads_q_params[head_idx]
        k_params = self.heads_k_params[head_idx]
        v_params = self.heads_v_params[head_idx]
        cond_params = self.heads_cond_params[head_idx]
        
        # Create quantum circuits for this head
        quantum_kernel = self.create_quantum_kernel(head_idx)
        quantum_attention = self.create_quantum_attention(head_idx)
        
        outputs = []
        for b in range(batch_size):
            attn_b = []
            for i in range(seq_len):
                scores = []
                for j in range(seq_len):
                    x_q = inputs[b, i, :]
                    x_k = inputs[b, j, :]
                    score = quantum_kernel(q_params, k_params, x_q, x_k)[0]
                    scores.append(score)
                
                scores = torch.stack(scores) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
                probs = torch.softmax(scores, dim=0)
                
                weighted = torch.zeros(self.n_qubits - 1, dtype=inputs.dtype, device=inputs.device)
                for j in range(seq_len):
                    x_q = inputs[b, i, :]
                    x_k = inputs[b, j, :]
                    x_v = inputs[b, j, :]
                    v_out = torch.stack(quantum_attention(q_params, k_params, v_params, cond_params, x_q, x_k, x_v))
                    weighted += probs[j] * v_out
                
                attn_b.append(self.head_out_linears[head_idx](weighted))
            outputs.append(torch.stack(attn_b))
        
        return torch.stack(outputs)

    def forward(self, inputs):
        """
        Forward pass through multihead attention.
        
        Args:
            inputs: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Process each head in parallel (conceptually - sequential in implementation)
        head_outputs = []
        for head_idx in range(self.num_heads):
            logger.info(f"Processing head {head_idx + 1}/{self.num_heads}")
            head_out = self.process_head(head_idx, inputs)
            head_outputs.append(head_out)
        
        # Concatenate all heads along the feature dimension
        # Each head output: (batch, seq_len, head_dim)
        multihead_output = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, d_model)
        
        # Final projection
        output = self.out_proj(multihead_output)
        
        return output
