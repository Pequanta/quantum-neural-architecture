import pennylane as qml
import torch
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quantum device (classical simulator)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)
# Hybrid QuantumAttention class
class QuantumAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_params = nn.Parameter(torch.randn(6))  # 2 wires * 3
        self.k_params = nn.Parameter(torch.randn(6))  # 2 wires * 3
        self.v_params = nn.Parameter(torch.randn(12)) # 4 wires * 3
        self.cond_params = nn.Parameter(torch.randn(9)) # 3 wires * 3
        self.out_linear = nn.Linear(n_qubits - 1, d_model)

    # Variational ansatz
    @staticmethod
    def variational_layer(params, wires):
        num_wires = len(wires)
        for i in range(num_wires):
            qml.RX(params[i * 3], wires=wires[i])
            qml.RY(params[i * 3 + 1], wires=wires[i])
            qml.RZ(params[i * 3 + 2], wires=wires[i])
        for i in range(num_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    # Encoding
    @staticmethod
    def encode_data(x, wires):
        for i, wire in enumerate(wires):
            qml.RY(x[i % len(x)], wires=wire)

    # Quantum kernel for scores
    @staticmethod
    @qml.qnode(dev, interface='torch')
    def quantum_kernel(q_params, k_params, x_q, x_k):
        QuantumAttention.encode_data(x_q, wires=range(n_qubits // 2))
        QuantumAttention.variational_layer(q_params, wires=range(n_qubits // 2))
        
        QuantumAttention.encode_data(x_k, wires=range(n_qubits // 2, n_qubits))
        qml.adjoint(QuantumAttention.variational_layer)(k_params, wires=range(n_qubits // 2, n_qubits))
        
        return qml.probs(wires=range(n_qubits))

    # Full quantum attention
    @staticmethod
    @qml.qnode(dev, interface='torch')
    def quantum_attention(q_params, k_params, v_params, cond_params, x_q, x_k, x_v):
        score = QuantumAttention.quantum_kernel(q_params, k_params, x_q, x_k)[0]
        
        QuantumAttention.encode_data(x_v, wires=range(n_qubits))
        QuantumAttention.variational_layer(v_params, wires=range(n_qubits))
        
        ancilla = n_qubits - 1
        qml.RY(score * torch.pi, wires=ancilla)
        qml.ctrl(QuantumAttention.variational_layer, control=ancilla)(cond_params, wires=range(n_qubits - 1))
        
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits - 1)]


    def forward(self, inputs):  # inputs: (batch, seq_len, d_model)
        batch_size, seq_len, _ = inputs.shape
        outputs = []
        for b in range(batch_size):  # Batch loop (for simplicity; optimize for production)
            attn_b = []
            for i in range(seq_len):
                scores = []
                for j in range(seq_len):
                    x_q = inputs[b, i, :]
                    x_k = inputs[b, j, :]
                    score = QuantumAttention.quantum_kernel(q_params=self.q_params, k_params=self.k_params, x_q=x_q, x_k=x_k)[0]
                    scores.append(score)
                scores = torch.stack(scores) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
                probs = torch.softmax(scores, dim=0)
                logger.info(f"Batch {b}, Position {i}: Softmax scores = {probs}")
                weighted = torch.zeros(n_qubits - 1, dtype=inputs.dtype, device=inputs.device)
                for j in range(seq_len):
                    x_k = inputs[b, j, :]
                    x_v = inputs[b, j, :]
                    v_out = torch.stack(QuantumAttention.quantum_attention(self.q_params, self.k_params, self.v_params, self.cond_params, x_q, x_k, x_v))
                    weighted += probs[j] * v_out
                attn_b.append(self.out_linear(weighted))
            outputs.append(torch.stack(attn_b))
        return torch.stack(outputs)


