"""Pure game rules shared by the API, search, and tests."""
from functools import lru_cache
import numpy as np

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def winning_line(board: list[list[int]], row: int, col: int, length: int,
                 rule: str = "freestyle") -> list[list[int]]:
    n, player = len(board), board[row][col]
    if not player:
        return []
    for dr, dc in DIRECTIONS:
        line = [[row, col]]
        ends = []
        for sign in (-1, 1):
            r, c = row + dr * sign, col + dc * sign
            part = []
            while 0 <= r < n and 0 <= c < n and board[r][c] == player:
                part.append([r, c])
                r, c = r + dr * sign, c + dc * sign
            ends.append(0 <= r < n and 0 <= c < n and board[r][c] == -player)
            line = part[::-1] + line if sign == -1 else line + part
        if len(line) >= length and (rule == "freestyle" or len(line) == length):
            if rule != "blocked" or not all(ends):
                return line
    return []


@lru_cache(maxsize=128)
def segments(n: int, length: int) -> np.ndarray:
    lines = []
    for r in range(n):
        for c in range(n):
            for dr, dc in DIRECTIONS:
                if 0 <= r + (length - 1) * dr < n and 0 <= c + (length - 1) * dc < n:
                    lines.append([(r + i * dr) * n + c + i * dc for i in range(length)])
    return np.asarray(lines, dtype=np.int64)


def tactical_scores(board: np.ndarray, player: int, length: int) -> np.ndarray:
    """Vectorized open-segment teacher; a heuristic, not the neural model."""
    n = board.shape[0]
    flat = board.ravel()
    idx = segments(n, length)
    windows = flat[idx]
    own, other = (windows == player).sum(1), (windows == -player).sum(1)
    weights = np.array([0.0, 2.0, 18.0, 180.0, 20000.0, 2000000.0, 200000000.0])
    # Scale relative to the requested winning length, including 3x3 and 4x4.
    weights[length - 1] = 1e7
    attack = np.where(other == 0, weights[own], 0)
    defend = np.where(own == 0, weights[other] * .96, 0)
    scores = np.zeros(n * n, dtype=np.float64)
    np.add.at(scores, idx.ravel(), np.repeat(attack + defend, length) * (windows == 0).ravel())
    rr, cc = np.indices((n, n))
    scores += (1 / (1 + np.abs(rr - (n - 1) / 2) + np.abs(cc - (n - 1) / 2))).ravel()
    scores[flat != 0] = -1
    return scores.reshape(n, n)
