#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from collections import namedtuple
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from enum import Enum, auto
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

Outcome = namedtuple("Outcome", "p00 p01 p10 p11")
Experiment = namedtuple("Experiment", "input_state outcome")


# ## Model Design
# 
# We will design a model that can predict the output probabilities given an input circuit design. Ideally, however, we could do the reverse problem: given a set of outcomes for all possible qubit inputs (00, 01, 10, 11), what is the circuit that creates it?
# 
# In order to accomplish either of these tasks, however, we need some vector representation of a quantum circuit design to feed to our model.

# In[2]:


single_qubit_gates = ["h", "iden", "u3", "u2", "u1", "rx", "ry", "rz", "x", "y", "z", "s", "sdg", "t", "tdg"]
double_qubit_gates_directed = ["cx", "ch", "cy", "cz", "crz", "cu1", "cu3"]
double_qubit_gates_undirected = ["swap"]

idx2gate = {}
idx = 0
# Single qubit gates can be in the first qubit, in the second qubit, or in both qubits
for qubit in (0, 1):
    for gate in single_qubit_gates:
        idx2gate[idx] = (gate, qubit)
        idx += 1
        
# Double qubit gates that have a direction
for control_qubit in (0, 1):
    target_qubit = 1 if control_qubit == 0 else 0
    for gate in double_qubit_gates_directed:
        idx2gate[idx] = (gate, control_qubit, target_qubit)
        idx += 1
        
# Double qubit gates without a direction
for gate in double_qubit_gates_undirected:
    idx2gate[idx] = (gate,)

idx2gate


# We will represent our circuit as a 44x10 matrix, allowing for 10 steps.

# In[3]:


# Choose between the following choices:
#   - single qubit gate at 0
#   - single qubit gate at 1
#   - single qubit gates at 0 and 1
#   - double qubit gate with control at 0
#   - double qubit gate with control at 1
#   - undirected double qubit gate (swap)

class GateOptions(Enum):
    SINGLE_QUBIT_GATE_AT_0 = auto()
    SINGLE_QUBIT_GATE_AT_1 = auto()
    SINGLE_QUBIT_GATE_AT_BOTH = auto()
    DOUBLE_QUBIT_GATE_WITH_CONTROL_AT_0 = auto()
    DOUBLE_QUBIT_GATE_WITH_CONTROL_AT_1 = auto()
    DOUBLE_QUBIT_GATE_UNDIRECTED = auto()
    
MAX_CIRCUIT_DEPTH = 10

def generate_random_circuit_step(idx2gate):
    step_vector = np.zeros(len(idx2gate))
    gate_option = random.choice(list(GateOptions))
    if gate_option == GateOptions.SINGLE_QUBIT_GATE_AT_0:
        idx = random.choice(np.arange(len(single_qubit_gates)))
        step_vector[idx] = 1

    elif gate_option == GateOptions.SINGLE_QUBIT_GATE_AT_1:
        idx = random.choice(np.arange(len(single_qubit_gates), len(single_qubit_gates) * 2))
        step_vector[idx] = 1

    elif gate_option == GateOptions.SINGLE_QUBIT_GATE_AT_BOTH:
        idx = random.choice(np.arange(len(single_qubit_gates)))
        step_vector[idx] = 1

        idx = random.choice(np.arange(len(single_qubit_gates), len(single_qubit_gates) * 2))
        step_vector[idx] = 1

    elif gate_option == GateOptions.DOUBLE_QUBIT_GATE_WITH_CONTROL_AT_0:
        start = len(single_qubit_gates) * 2
        idx = random.choice(np.arange(start, start + len(double_qubit_gates_directed)))
        step_vector[idx] = 1

    elif gate_option == GateOptions.DOUBLE_QUBIT_GATE_WITH_CONTROL_AT_1:
        start = len(single_qubit_gates) * 2 + len(double_qubit_gates_directed)
        idx = random.choice(np.arange(start, start + len(double_qubit_gates_directed)))
        step_vector[idx] = 1

    elif gate_option == GateOptions.DOUBLE_QUBIT_GATE_UNDIRECTED:
        start = len(single_qubit_gates) * 2 + len(double_qubit_gates_directed) * 2
        idx = random.choice(np.arange(start, start + len(double_qubit_gates_undirected)))
        step_vector[idx] = 1
        
    return step_vector

def generate_random_circuit_matrix(idx2gate):
    depth = random.choice(range(1, MAX_CIRCUIT_DEPTH + 1))
    step_vectors = []
    for _ in range(depth):
        step_vector = generate_random_circuit_step(idx2gate)
        step_vectors.append(step_vector)
    # Add padding
    step_vector_size = len(step_vectors[0])
    padding_amount = MAX_CIRCUIT_DEPTH - depth
    padding = padding_amount * [step_vector_size * [0]]
    circuit_matrix = np.array(step_vectors + padding)
    return circuit_matrix


# In[4]:


def find_outcomes(circuit_matrix, idx2gate):
    """Find the outcomes of all different qubit inputs.
    
    Qubit inputs are always in the ground state, so we have to
    apply a SWITCH gate (x) in order to flip the Qubit to change
    the input.
    """
    experiments = []
    
    # 00
    circuit_00 = QuantumCircuit(2, 2)
    create_circuit_from_matrix(circuit_00, circuit_matrix, idx2gate)
    experiment = Experiment("00", run_simulation(circuit_00))
    experiments.append(experiment)
    
    # 01
    circuit_01 = QuantumCircuit(2, 2)
    circuit_01.x(0)
    create_circuit_from_matrix(circuit_01, circuit_matrix, idx2gate)
    experiment = Experiment("01", run_simulation(circuit_01))
    experiments.append(experiment)
    
    # 10
    circuit_10 = QuantumCircuit(2, 2)
    circuit_10.x(1)
    create_circuit_from_matrix(circuit_10, circuit_matrix, idx2gate)
    experiment = Experiment("10", run_simulation(circuit_10))
    experiments.append(experiment)
    
    # 11
    circuit_11 = QuantumCircuit(2, 2)
    circuit_11.x(0)
    circuit_11.x(1)
    create_circuit_from_matrix(circuit_11, circuit_matrix, idx2gate)
    experiment = Experiment("11", run_simulation(circuit_11))
    experiments.append(experiment)
    
    return circuit_00, experiments
    
def run_simulation(circuit, shots=2000):
    circuit.measure([0,1], [0,1])
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    
    p00 = counts.get("00", 0) / shots
    p01 = counts.get("01", 0) / shots
    p10 = counts.get("10", 0) / shots
    p11 = counts.get("11", 0) / shots
    
    return Outcome(p00, p01, p10, p11)

def create_circuit_from_matrix(circuit, circuit_matrix, idx2gate):
    random_degree = lambda: 45  # random.random() * 360
    for step_vector in circuit_matrix:
        for gate_idx in np.where(step_vector == 1)[0]:
            gate_name, *args = idx2gate[gate_idx]
            gate_fn = eval(f"circuit.{gate_name}")
            if not args:
                # Double qubit gate undirected (swap)
                gate_fn(0, 1)
                
            elif len(args) == 1:
                # Single qubit gate
                qubit = args[0]
                if gate_name in {"u1", "rx", "ry"}:
                    theta = random_degree()
                    gate_fn(theta, qubit)
                elif gate_name == "u3":
                    theta = random_degree()
                    phi = random_degree()
                    lam = random_degree()
                    gate_fn(theta, phi, lam, qubit)
                elif gate_name == "rz":
                    phi = random_degree()
                    gate_fn(phi, qubit)
                elif gate_name == "u2":
                    lam = random_degree()
                    phi = random_degree()
                    gate_fn(phi, lam, qubit)
                else:
                    gate_fn(qubit)
                
            elif len(args) == 2:
                # Double qubit gate
                control, target = args
                if gate_name in {"cu1", "crz"}:
                    theta = random_degree()
                    gate_fn(theta, control, target)
                elif gate_name == "cu3":
                    theta = random_degree()
                    phi = random_degree()
                    lam = random_degree()
                    gate_fn(theta, phi, lam, control, target)
                else:
                    gate_fn(control, target)


# In[5]:


button = widgets.Button(description="Show example")
output = widgets.Output()
display(button, output)

def show_generated_training_example(button):
    circuit_matrix = generate_random_circuit_matrix(idx2gate)
    circuit, experiments = find_outcomes(circuit_matrix, idx2gate)
    with output:
        clear_output(wait=True)
        for experiment in experiments:
            print("Input qubits:", experiment.input_state)
            print("Outcome:")
            outcome = experiment.outcome
            print(f"{outcome.p00:.2f}\t{outcome.p01:.2f}\t{outcome.p10:.2f}\t{outcome.p11:.2f}")
            print()
        display(circuit.draw(output="mpl"))

button.on_click(show_generated_training_example)


# In[365]:


class Examples(Dataset):
    def __getitem__(self, _):
        circuit_matrix = generate_random_circuit_matrix(idx2gate)
        _, experiments = find_outcomes(circuit_matrix, idx2gate)
        outcomes = [experiment.outcome for experiment in experiments]
        outcomes = [[outcome.p00, outcome.p01, outcome.p10, outcome.p11] for outcome in outcomes]
        flattened_outcomes = np.array([sublst for lst in outcomes for sublst in lst])
        return circuit_matrix, flattened_outcomes
    
    def __len__(self):
        # "Infinite" training examples
        return int(1e10)


# In[366]:


dataset = Examples()
BATCH_SIZE = 512
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)


# In[367]:


class CircuitPredictor(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        batch_size,
        hidden_size=128, 
        n_layers=1,
        device='cpu',
    ):
        super(CircuitPredictor, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size
        self.output_size = output_size
        
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)
    
    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        output, hidden = self.rnn(inputs.float(), self.init_hidden())
        output = self.decoder(output[:, -1, :]).squeeze()
        return F.sigmoid(output)


# In[368]:


model = CircuitPredictor(len(idx2gate), 4 * 4, BATCH_SIZE)
model


# In[369]:


criterion = nn.MSELoss()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)


# In[370]:


losses = []
for i, (circuit_matrix, outcomes) in enumerate(train_loader, 1):
    model.zero_grad()
    output = model(circuit_matrix)
    loss = criterion(output, outcomes.float())
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.3f}")
    
    losses.append(loss.item())


# In[ ]:




