from torch import nn
import torch
import pennylane as qml
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
class VQCLayer(nn.Module):
    def __init__(self, num_qubits=6, n_layers=12, backend='default.qubit', encoder=None):
        super().__init__()
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.encoder = encoder
        self.dev = qml.device(backend, wires=num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # first embedding (unchanged)
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits))

            # data re-uploading inside each block
            for l in range(self.n_layers):
                # parametrized single-qubit rotations
                for q in range(self.num_qubits):
                    qml.RY(weights[l, q, 0], q)
                    qml.RZ(weights[l, q, 1], q)

                # re-encode inputs (gives frequency lift)
                qml.AngleEmbedding(inputs, wires=range(self.num_qubits))

                # ring entanglement (shallow, local)
                for q in range(self.num_qubits):
                    qml.CZ(wires=[q, (q + 1) % self.num_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

        # better weight init scale
        weight_shape = (self.n_layers, self.num_qubits, 2)  # we only use 2 params/qubit now
        std = 1 / np.sqrt(num_qubits)    # xavier-ish
        self.weights = nn.Parameter(std * torch.randn(weight_shape))

    def forward(self, input_):
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)
        return torch.stack([torch.stack(self.circuit(x, self.weights)) for x in input_])
