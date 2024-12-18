import torch
from torch import nn

import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Define LSTM and fully connected layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Register hidden and cell state buffers for TorchScript compatibility
        self.register_buffer("hidden_state", torch.zeros(self.num_layers, 1, self.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x):
        # Use hidden state and cell state from buffers
        hidden = (self.hidden_state, self.cell_state)

        # Forward pass through LSTM
        out, (h_n, c_n) = self.lstm(x, hidden)

        # Update hidden and cell states
        self.hidden_state.copy_(h_n)
        self.cell_state.copy_(c_n)

        # Fully connected layer (output at the last timestep)
        out = self.fc(out[:, -1, :])
        return out

        
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.hidden = self.reset_hidden_state(1)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the output at the last time step
        return out, hidden

    def reset_hidden_state(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)