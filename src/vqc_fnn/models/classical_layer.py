import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np


class ClassicalLayer(nn.Module):
    """
    Classical layer for hybrid quantum-classical models.
    
    Features:
    1. Dimensionality reduction (PCA-based initialization).
    2. Post-processing of quantum outputs.
    3. Trainability enhancements (Non-linearity, BatchNorm).
    """
    def __init__(self, input_dim, n_qubits, output_dim, hidden_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else n_qubits * 2

        # 1. Dimensionality Reduction (PCA-based)
        # We use a Linear layer that can be initialized with PCA components.
        self.reduction_layer = nn.Linear(input_dim, n_qubits)

        # 2. Post-processing of quantum outputs & 3. Trainability enhancements
        self.post_process = nn.Sequential(
            nn.Linear(n_qubits, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(), # Non-linearity
            nn.Linear(self.hidden_dim, output_dim)
        )

    def fit_pca(self, data):
        """
        Computes PCA on the provided data and initializes the reduction layer weights.

        Args:
            data (torch.Tensor or array-like): Input data of shape (N, input_dim).
        """
        # Convert to numpy for sklearn
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data, dtype=np.float32)

        if data_np.ndim != 2 or data_np.shape[1] != self.input_dim:
            raise ValueError(f"Expected data of shape (N, {self.input_dim}); got {data_np.shape}")

        # Determine usable number of components
        num_components = min(self.n_qubits, self.input_dim, data_np.shape[0])

        # Fit PCA using sklearn
        pca = PCA(n_components=num_components)
        pca.fit(data_np)

        components = pca.components_  # shape (num_components, input_dim)
        mean = pca.mean_  # shape (input_dim,)

        with torch.no_grad():
            # copy components into the reduction layer weights (first num_components rows)
            comp_t = torch.from_numpy(components).to(dtype=self.reduction_layer.weight.dtype, device=self.reduction_layer.weight.device)
            self.reduction_layer.weight[:num_components, :].copy_(comp_t)

            # set corresponding bias to -components @ mean
            bias_part = -comp_t @ torch.from_numpy(mean).to(dtype=comp_t.dtype, device=comp_t.device)
            self.reduction_layer.bias[:num_components].copy_(bias_part)

        print(f"PCA initialized. Reduction layer weights set from top {num_components} components.")

    def forward(self, x, quantum_layer=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input data (batch_size, input_dim).
            quantum_layer (nn.Module, optional): Quantum layer to process reduced data.
        
        Returns:
            torch.Tensor: Output of the classical post-processing.
        """
        # 1. Reduce dimensionality
        reduced_x = self.reduction_layer(x)
        
        # 2. Process with Quantum Layer (if provided)
        if quantum_layer is not None:
            # Depending on the quantum layer implementation, it might expect specific range or shape
            # Typically VQC layers might expect inputs in [0, 2pi] or similar if using AngleEmbedding
            # Here we assume the quantum layer handles its own embedding or the PCA output is suitable.
            # For AngleEmbedding, we might want to scale this, but we'll leave it to the user/quantum layer.
            q_out = quantum_layer(reduced_x)
            # Quantum layer might return float64 (Double), cast to float32
            if q_out.dtype != reduced_x.dtype:
                q_out = q_out.to(reduced_x.dtype)
        else:
            # Bypass quantum layer (for testing or classical-only baseline)
            q_out = reduced_x

        # 3. Post-process
        out = self.post_process(q_out)
        
        return out
