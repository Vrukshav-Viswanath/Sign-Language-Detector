"""
Two model architectures:
  - LetterMLP: classifies a single frame's hand landmarks -> letter
  - WordGRU:   classifies a sequence of frames' landmarks -> word
"""

import torch
import torch.nn as nn


class LetterMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):  # x: (batch, input_dim)
        return self.net(x)


class WordGRU(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # final timestep's hidden state (both directions)
        return self.head(last)