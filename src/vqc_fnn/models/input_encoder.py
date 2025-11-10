import pennylane as qml
import numpy as np
from matplotlib import pyplot as plt
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

if __name__ == "__main__":
    # Define input and device
    inst = InputEncoder(np.asarray([0, 0, 1]), 3)
    dev = qml.device('default.qubit', wires=3)
    circuit = inst.encode_input(dev)

    another_circuit = inst.encode_input(dev, operation_type="operation", with_padding=True, operations_list=[qml.Hadamard(wires=0), qml.CNOT(wires=[0,1]), qml.CNOT(wires=[1,2])])
    try:
        result = another_circuit()  # Execute the circuit
        print("Expectation value <Z_0>:", result)
        print("Input after padding:", inst.classic_input)
    except Exception as e:
        print("Error:", e)
    fig, ax = qml.draw_mpl(another_circuit)()
    plt.show()
