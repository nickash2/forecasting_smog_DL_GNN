# src/modelling/MLP.py

# Class definition for the Multi-Layer Perceptron (MLP) model

import torch
import torch.nn as nn

class BasicMLP(nn.Module):
    """
    Defines an MLP network, with the member functions:
    - __init__: constructor
    - forward: defines the forward pass of the MLP
    """

    def __init__(self, N_INPUT_UNITS, N_HIDDEN_LAYERS, N_HIDDEN_UNITS, N_OUTPUT_UNITS):
        super(BasicMLP, self).__init__()
        self.layers = nn.Sequential(*[  # init input layer, hidden layers, and output layer
            nn.Linear(N_INPUT_UNITS, N_HIDDEN_UNITS),
            nn.ReLU(),
            *[
                nn.Linear(N_HIDDEN_UNITS, N_HIDDEN_UNITS),
                nn.ReLU(),
            ] * N_HIDDEN_LAYERS,
            nn.Linear(N_HIDDEN_UNITS, N_OUTPUT_UNITS),
        ])

    def forward(self, u):               # u: (batch_size, seq_length, n_features)
                                        # outputs: (batch_size, seq_length, N_OUTPUT_UNITS)
        batch_size, seq_length, n_features = u.size()
        outputs = torch.empty((batch_size, seq_length, self.layers[-1].out_features))

        for t in range(seq_length):     # iterate over sequence length
            u_t = u[:, t, :]            # u_t: (batch_size, n_features)
            y_t = self.layers(u_t)      # y_t: (batch_size, N_OUTPUT_UNITS)
            outputs[:, t, :] = y_t      # store the output and return last 24 hours
        return outputs[:, -24:, :]