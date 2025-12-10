# baselines.py
import torch
from torch import nn
import math

n_q = 6  # compressed input dimension (match number of qubits)

# -------------------------
# A) Parameter-matched MLP (6 -> 2 -> 6) => 32 params
# -------------------------
class ParamMatchedMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=20, output_dim=3, activation='tanh'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh() if activation=='tanh' else nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class RFFSmallMLP(nn.Module):
    def __init__(self, n_q=6, m=6, hidden=6, train_w=False, scale=10.0):
        """
        n_q: input compressed dimension
        m: number of random Fourier frequencies (results in 2*m features)
        hidden: hidden layer size for small MLP
        train_w: whether to learn projection matrix W (if True -> more params)
        scale: scale of random frequencies
        """
        super().__init__()
        self.n_q = n_q
        self.m = m
        self.hidden = hidden

        # Random projection: W (m x n_q) and b (m)
        W = scale * torch.randn(m, n_q)
        b = 2 * math.pi * torch.rand(m)
        if train_w:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer('W', W)
            self.register_buffer('b', b)

        # small MLP mapping 2*m -> hidden -> n_q (or to classes)
        self.fc1 = nn.Linear(2 * m, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, n_q, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        # x: (batch, n_q)
        # compute z = W x^T + b  -> shape (batch, m)
        z = torch.matmul(x, self.W.t()) + self.b  # (batch, m)
        # RFF features:
        rff = torch.cat([torch.sin(z), torch.cos(z)], dim=1)  # (batch, 2*m)
        h = self.act(self.fc1(rff))
        out = self.fc2(h)  # (batch, n_q)
        return out
