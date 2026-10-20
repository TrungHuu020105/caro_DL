"""A fully convolutional policy/value CNN. Works with every supported board size."""
import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class CaroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(5, 24, 3, padding=1), nn.ReLU())
        self.tower = nn.Sequential(ResidualBlock(24), ResidualBlock(24), ResidualBlock(24))
        self.policy = nn.Sequential(nn.Conv2d(24, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.value = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(24, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        features = self.tower(self.stem(x))
        logits = self.policy(features).flatten(1)
        logits = logits.masked_fill(x[:, 2].flatten(1) < .5, -1e9)
        return logits, self.value(features).squeeze(1)


def encode(board: np.ndarray, player: int, length: int) -> np.ndarray:
    n = board.shape[0]
    return np.stack([(board == player), (board == -player), (board == 0),
                     np.full((n, n), length / 6), np.ones((n, n))]).astype(np.float32)
