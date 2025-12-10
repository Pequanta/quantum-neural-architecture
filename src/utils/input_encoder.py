import pennylane as qml
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

np.random.seed(11111)

class InputEncoder:
    def __init__(self, classic_input: np.ndarray , n_qubits: int, device_type = 'default.qubit'):
        self.classic_input = classic_input
        self.n_qubits = n_qubits
        self.device = qml.device(device_type, wires=n_qubits)
        self.random_generator = np.random.default_rng()
        self.circuit = None
    def add_padding(self):
        """"
        Add padding to the input data to match the number of qubits. 
        This is done by adding zeros to the end of the input data to make the length
        the power of 2.
        """
        length = len(self.classic_input)
        next_power_of_2 = 2 ** np.ceil(np.log2(length)).astype(int)
        padding_length = next_power_of_2 - length
        self.classic_input = np.pad(self.classic_input, (0, padding_length), 'constant')


    def prepare_state(self):
        """
        Used to prepare the quantum state from classical input data.
        1. Add padding to the input data to match the number of qubits.
        2. Apply Hadamard gate to each qubit to create superposition state.
        3. Return the prepared quantum state.
        """
        # Apply Hadamard gate to each qubit to create superposition state
        for i in range(len(self.classic_input)):
            qml.Hadamard(wires=i)

    def encode_input(self, device, gate_type='Y', embedding_type = 'angle', operation_type = "initialization", with_padding=True, operations_list = None):
        @qml.qnode(device)
        def circuit():
            if operation_type == "initialization":  
                if with_padding:
                    self.add_padding()
                if embedding_type == 'angle':
                    self.prepare_state()
                    #creating an angle embedding for the input data
                    qml.AngleEmbedding(self.classic_input, wires=range(len(self.classic_input)), rotation=gate_type)
                elif embedding_type == 'amplitude':
                    #creating an amplitude embedding for the input data
                    qml.AmplitudeEmbedding(self.classic_input, wires=range(len(self.classic_input)), normalize=True, pad_with=0.0)
            elif operations_list:
                for i in range(len(operations_list)):
                    qml.apply(operations_list[i])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit
        return self.circuit

class InputPreprocessing:
    """
    Class to preprocess the dataset input into the form usable with the current architecture.
    """
    def __init__(self, data, target, test_size=0.3, random_state=42):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def preprocess(self):
        """
        Preprocesses the data:
        1. Normalizes features.
        2. Splits into train and test sets.
        3. Converts to PyTorch tensors.
        
        Returns:
            X_train, X_test, y_train, y_test (torch.Tensor)
        """
        # Normalize features
        X = self.scaler.fit_transform(self.data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.target, test_size=self.test_size, random_state=self.random_state
        )
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        return X_train, X_test, y_train, y_test

