import warnings

import torch
import torch.nn as nn


class BidirectionalGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        """Initialize a bidirectional GRU

        Args:
            n_in (int): number of expected input features
            n_hidden (int): number of hidden state features
            dropout (float, optional): dropout. Defaults to 0.
            num_layers (int, optional): number of recurrent layers. Defaults to 1.
        """
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_in,
            hidden_size=n_hidden,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        """Apply the Bidirectional GRU to a given tensor

        Args:
            x (tensor):
        Returns:
            out (tensor)
        """
        out, _ = self.rnn(x)
        return out
