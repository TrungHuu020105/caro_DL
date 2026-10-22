"""Rule-conditioned policy/value CNN with masked pooling for variable boards."""
import numpy as np
import torch
from torch import nn

RULES = ("freestyle", "exact", "blocked")
INPUT_CHANNELS = 8
MODEL_VERSION = 2


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
        self.stem = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, 24, 3, padding=1), nn.ReLU())
        self.tower = nn.Sequential(ResidualBlock(24), ResidualBlock(24), ResidualBlock(24))
        self.policy = nn.Sequential(nn.Conv2d(24, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.value = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        features = self.tower(self.stem(x))
        logits = self.policy(features).flatten(1)
        mask = x[:, 4:5]
        logits = logits.masked_fill((x[:, 2] * x[:, 4]).flatten(1) < .5, -1e9)
        pooled = (features * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1)
        return logits, self.value(pooled).squeeze(1)


def encode(board: np.ndarray, player: int, length: int, rule: str = "freestyle",
           canvas_size: int | None = None) -> np.ndarray:
    n = board.shape[0]
    if rule not in RULES or canvas_size is not None and canvas_size < n:
        raise ValueError("Invalid rule or canvas size.")
    encoded = np.stack([(board == player), (board == -player), (board == 0),
                        np.full((n, n), length / 6), np.ones((n, n)),
                        *[np.full((n, n), rule == name) for name in RULES]]).astype(np.float32)
    if canvas_size is None or canvas_size == n:
        return encoded
    canvas = np.zeros((INPUT_CHANNELS, canvas_size, canvas_size), dtype=np.float32)
    canvas[:, :n, :n] = encoded
    return canvas
