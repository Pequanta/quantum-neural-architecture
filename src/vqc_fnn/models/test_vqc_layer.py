
import torch
from torch import nn, optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pennylane as qml
from input_encoder import InputEncoder  # your encoder class
from vqc_layer import VQCLayer          # the quantum layer we defined earlier
from matplotlib import pyplot as plt
# -----------------------------
# 1. Data Preparation
# -----------------------------
iris = datasets.load_iris()
X = iris.data  # 4 features per sample
y = iris.target

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

loss_history = []          # <--- add
acc_history  = []          # <--- add

# Convert labels to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# 2. Define the Hybrid Model
# -----------------------------
class HybridVQCFNN(nn.Module):
    def __init__(self, num_qubits=4, n_layers=2):
        super().__init__()
        self.fc1 = nn.Linear(4, num_qubits)  # classic preprocessing
        self.encoder = InputEncoder(np.zeros(num_qubits), n_qubits=num_qubits)
        self.q_layer = VQCLayer(num_qubits=num_qubits, n_layers=n_layers, encoder=self.encoder)
        self.fc2 = nn.Linear(num_qubits, 3)  # 3 output classes for Iris

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.q_layer(x)
        x = x.to(torch.float32)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

model = HybridVQCFNN(num_qubits=6, n_layers=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-3 
)
# -----------------------------
# 3. Training Loop
# -----------------------------
EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        acc = (preds == y_test).float().mean()
    loss_history.append(loss.item())          # <--- add
    acc_history.append(acc.item()) 
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Test Acc: {acc.item()*100:.2f}%")

# -----------------------------
# 4. Final Evaluation
# -----------------------------
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)
    acc = (preds == y_test).float().mean()
print("\nFinal Test Accuracy:", acc.item()*100, "%")


# -----------------------------
# 4. Plot learning curves
# -----------------------------
plt.figure(figsize=(8,3))
plt.plot(loss_history)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure(figsize=(8,3))
plt.plot([a*100 for a in acc_history])
plt.title("Test Accuracy (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()